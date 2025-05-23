import torch
import nnsight
import time

from typing import Literal
from typing import Any, Callable, Literal

from tqdm import tqdm
from huggingface_hub import hf_hub_download
from nnsight import LanguageModel 
from nnsight.intervention.base import InterventionProxy
from transformers import BitsAndBytesConfig
from sae_lens import SAE

SAE_MAPPER = {
    "meta-llama/Llama-3.1-8B-Instruct":  "Llama-3.1-8B-Instruct-SAE-l{layer}",
    "google/gemma-2-2b-it": "gemma-scope-2b-pt-res-canonical",
    "google/gemma-2-9b-it": "gemma-scope-9b-pt-res-canonical",
}



class SAEWrapper:
    def __init__(
        self,
        sae_name: str,
        sae_layer: int,
        device: torch.device | None = None,
    ):
        self.sae_type = Literal['goodfire', 'gemma-scope']
        self.device = device

        if sae_name == "Llama-3.1-8B-Instruct-SAE-l19":
            self.sae_type = "goodfire"
            assert sae_layer == 19, "SAE layer must be 19 for goodfire SAE"

            sae_name = "Llama-3.1-8B-Instruct-SAE-l19"
            file_path = hf_hub_download(
                repo_id=f"Goodfire/{sae_name}",
                filename=f"{sae_name}.pth",
                repo_type="model",
            )

            self.sae = load_goodfire_sae(
                file_path,
                d_model=4096,
                expansion_factor=16,
                device=device,
            )
            self.sae = self.sae.to(device)
            self.dtype = self.sae.dtype

            self.d_in = self.sae.d_in 
            self.d_hidden = self.sae.d_hidden 
        elif 'scope' in sae_name:
            # eg: "gemma-scope-2b-pt-res-canonical"
            self.sae_type = "gemma-scope"

            self.sae, cfg_dict, _ = SAE.from_pretrained(
                release = sae_name,
                sae_id = f"layer_{sae_layer}/width_16k/canonical",
            )
            self.sae = self.sae.to(device)
            self.dtype = self.sae.dtype

            self.d_in = cfg_dict["d_in"]
            self.d_hidden = cfg_dict["d_sae"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return self.sae.encode(x.to(self.device).to(self.sae.dtype))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.sae.decode(x.to(self.device).to(self.sae.dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x.to(self.device).to(self.sae.dtype))
        return self.decode(f), f

class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device | None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f


def load_goodfire_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device | None = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)

    return sae



class NNsightWrapper:
    def __init__(self, model_name, load_in_4bit=False):
        
        self._model = LanguageModel(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        # Quickly run a trace to force model to download due to nnsight lazy download
        input_tokens = self._model.tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])
        with self._model.trace(input_tokens):
            pass

        self.tokenizer = self._model.tokenizer
        self.config = self._model.config
        self.safe_mode = False  # Nnsight validation is disabled by default, slows down inference a lot. Turn on to debug.
        self.device = self._model.device
        if self._model.config and hasattr(self._model.config, "hidden_size"):
            self.d_model: int = int(self._model.config.hidden_size)
        else:
            raise Exception("Model does not have hidden_size attribute.")


    def _find_module(self, hook_point):
        submodules = hook_point.split('.')
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module


    # se devi settare le features, passa alla forward la funzione partial con già il sae caricato 
    # che poi verrà chiamato runtime durante l'esecuzione passando le activations per calcolare quelle nuove
    def set_forward(
        self, 
        inp_tokens: torch.Tensor,
        interventions: dict[str, Callable[[InterventionProxy], torch.Tensor]],    # {hook_point: f(act) -> new_act)}
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.2,
        set_max_tokens_to_input_size: bool = True,
        return_new_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Function to run the steered model according to the `interventions`'s functions.
        Args:
            inp_tokens (torch.Tensor): The input tokens to use.
            interventions (dict[str, Callable[[InterventionProxy], torch.Tensor]]): Dict containing hook_point(s) as key(s) and intervention functions as value. Such functions take the activations as input and return the steered activations.
            max_new_tokens (int): The maximum number of tokens to generate.
        Returns:
            torch.Tensor: The output tokens after applying the interventions.
        """
        inp_tokens = inp_tokens.to(self._model.device)
        inp_len = inp_tokens.shape[-1]

        with self._model.generate(
            inp_tokens,
            max_new_tokens=int(inp_len * 1.5) if set_max_tokens_to_input_size else max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            do_sample=False,
        ) as _:
            # nnsight list
            new_activations_list = nnsight.list().save()

            for hook_point in interventions.keys():
                module = self._find_module(hook_point)
                with module.all():      # all pathes at each generation step the same maodule
                    intervention_values = nnsight.apply(
                        interventions[hook_point],
                        activations=module.output[0][:, -1:, :],       # shape: (batch_size, inp_len if first_step else 1 (current tok) - getting the last tok keeping the dim (-1: ) whatever the case, d_model)
                    )
                    # Apply intervention - set first layer output to zero
                    module.output[0][:, -1:, :] = intervention_values

                    new_activations_list.append(intervention_values)

            out = self._model.generator.output.save()

        if return_new_activations:
            return out[0, inp_len:], new_activations_list
        return out[0, inp_len:]



    def cache_forward(
        self,
        inp_tokens: torch.Tensor,
        hook_points: list[str],      # hook points
        max_new_tokens: int = 512,
        return_out_tokens: bool = False,
    ) -> list[dict[str, InterventionProxy | Any]]:
        inp_tokens = inp_tokens.to(self._model.device)

        with self._model.generate(
            inp_tokens, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
        ) as _:
            activations: list[                          # for every gen token
                dict[str, InterventionProxy | Any]      # hook_point: activations
            ] = nnsight.list().save()

            # old, not working when using multiple hook points
            for hook_point in hook_points:
                module = self._find_module(hook_point)
                with module.all():
                    activations.append(
                        {hook_point: module.output[0][:, -1:, :]}
                    )

            # for hook_point in hook_points:
            #     module = self._find_module(hook_point)
            #     with module.all():          # equivalent to `for _ in range(gen_tokens)`
            #         activations.append({})
            #         activations[-1][hook_point] = module.output[0][:, -1:, :]

            out = self._model.generator.output.save()

        clean_activations = []
        # (now e.g. activations = [{hook1: act}, {hook2: act}, {hook1: act}, {hook2: act}]) (2 gen tokens))
        n_hooks = len(hook_points)
        for i in range(0, len(activations), n_hooks):
            merged_dict = {}
            for j in range(n_hooks):
                activations[i + j][hook_points[j]] = activations[i + j][hook_points[j]][0, -1, :].detach().cpu()
                merged_dict.update(activations[i + j])
            clean_activations.append(merged_dict)
        # now e.g. activations = [{hook1: act, hook2: act}, {hook1: act, hook2: act}]

        if return_out_tokens:
            return clean_activations, out

        return clean_activations
        """
        Returns:
        ```python
        activations: list[                          # for every gen token
            {                                       # 1st gen token
                hook_point1: activations,
                hook_point2: activations,
            },
            {                                       # 2nd gen token
                hook_point1: activations,
                hook_point2: activations,
            },
        ]
        ```
        """



    def cache_forward_forced(
        self,
        inp_tokens: torch.Tensor,
        pe: torch.Tensor,
        mt: torch.Tensor,
        hook_points: list[str],      # hook points
        max_new_tokens: int = 512,
    ) -> tuple[
       list[dict[str, InterventionProxy | Any]],  # cache for every gen token {'hook_point': activations}
       list[dict[str, int]],  # for every pe token, what is generated, what is the MT token and what the PE token is
    ]:
        """
        **DEPRECATED**: Use cache_forward instead.

        Force the model to generate the PE tokens and record its activations.
        Args:
            inp_tokens (torch.Tensor): The input tokens to use.
            pe (torch.Tensor): The PE tokens to use. This will also correspond to the model's generation.
            mt (torch.Tensor): The MT tokens to use. This will be used to check if the model is following the PE or MT generation.
            hook_points (list[str]): The hook points to use for extracting features.
            max_new_tokens (int): The maximum number of tokens to generate.
        Returns:
            A tuple containing:
            ```python
            cache: list[                            # for every gen token
                dict[str, InterventionProxy | Any]  # hook_point: activations
            ]
            tokens_record: list[dict[str, int]]     # for every generated (pe) token, {"gen": gen_token, "pe": pe_token (reference), "mt": mt_token}
            ```
        """
        inp_tokens = inp_tokens.to(self._model.device)
        pe = pe.to(self._model.device)

        cache: list[                            # for every gen token
            dict[str, InterventionProxy | Any]   # hook_point: activations
        ] = []
        tokens_record: list[dict[str, int]] = list()
        
        for step in tqdm(range(max_new_tokens), desc="getting act", leave=False):
            # # Apparently the Trace + Invoke has some kind of memory leak, nnsight 0.4.5 -- waiting for a fix im gonna use the generate (less clean but still effective)
            # with self._model.trace(
            #         scan=self.safe_mode, validate=self.safe_mode
            #     ) as tracer:
            #     with tracer.invoke(inp_tokens) as _:

            with self._model.generate(inp_tokens, max_new_tokens=1, pad_token_id=self._model.tokenizer.eos_token_id) as _:
                # hook point here
                cache.append({})
                for hook_point in hook_points:
                    module = self._find_module(hook_point)
                    cache[step][hook_point] = module.output[0].save()   # save the activations
                        
                _vocab = self._model.lm_head.output.save()
            next_token = _vocab[0, -1, :].argmax(-1).detach()

            # nope, i'm forcing PE
            # if next_token.item() == self._model.tokenizer.eos_token_id:
            #     break

            tokens_record.append({
                "gen": next_token.item(),
                "pe": pe[0, step].item(),
                "mt": mt[0, step].item() if step < mt.shape[-1] else -1,
            })

            # detach and keep the last token activation
            cache[step] = {k: v[0, -1, :].detach().cpu() for k, v in cache[step].items()}

            if step < pe.shape[-1] - 1:     # current step is the last token of PE
                inp_tokens = torch.cat((
                    inp_tokens.detach(),
                    pe[:, step].unsqueeze(0),       # force generation on PE
                ), dim=-1)
            else:
                # no more PE tokens to use, the model is keeping generating stuff when it doesn't have to
                break

        return cache, tokens_record

