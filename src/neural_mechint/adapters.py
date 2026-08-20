from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .types import ActivationCache


@dataclass
class ForwardResult:
    logits: torch.Tensor
    cache: ActivationCache


class HFCausalLMAdapter:
    """Architecture-tolerant hook layer over Hugging Face causal LMs.

    Block outputs (the post-block residual stream) are used as the common
    observation/intervention surface across Gemma, Llama, Qwen, GPT-2,
    GPT-NeoX and closely related decoder-only architectures.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.blocks = self._resolve_blocks(model)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device_map: str | dict | None = "auto",
        torch_dtype: str | torch.dtype = "auto",
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "HFCausalLMAdapter":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the `frontier` extra: pip install -e '.[frontier]'") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        model.eval()
        return cls(model, tokenizer)

    @staticmethod
    def _resolve_blocks(model):
        candidates = [
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            ("model", "decoder", "layers"),
        ]
        for path in candidates:
            obj = model
            try:
                for name in path:
                    obj = getattr(obj, name)
                if len(obj) > 0:
                    return obj
            except (AttributeError, TypeError):
                continue
        raise ValueError(
            "Could not locate transformer blocks. Pass a supported decoder-only Hugging Face model "
            "or extend HFCausalLMAdapter._resolve_blocks for this architecture."
        )

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def tokenize(self, prompt: str) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in batch.items()}

    def single_token_id(self, text: str) -> int | None:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0]) if len(ids) == 1 else None

    @staticmethod
    def _hidden(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")

    @staticmethod
    def _replace_hidden(output, hidden: torch.Tensor):
        if isinstance(output, torch.Tensor):
            return hidden
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")

    def forward_with_cache(
        self,
        prompt: str,
        layers: Iterable[int] | None = None,
        *,
        detach_to_cpu: bool = True,
    ) -> ForwardResult:
        wanted = set(range(self.n_layers) if layers is None else layers)
        cache: dict[int, torch.Tensor] = {}
        handles = []

        for idx in sorted(wanted):
            if idx < 0 or idx >= self.n_layers:
                raise IndexError(f"layer {idx} outside [0, {self.n_layers})")

            def hook(_module, _inputs, output, layer_idx=idx):
                hidden = self._hidden(output).detach()
                cache[layer_idx] = hidden.cpu() if detach_to_cpu else hidden

            handles.append(self.blocks[idx].register_forward_hook(hook))

        batch = self.tokenize(prompt)
        try:
            with torch.inference_mode():
                logits = self.model(**batch).logits
        finally:
            for handle in handles:
                handle.remove()

        return ForwardResult(
            logits=logits,
            cache=ActivationCache(values=cache, tokens=batch.get("input_ids"), metadata={"prompt": prompt}),
        )

    def forward_with_patch(
        self,
        prompt: str,
        *,
        layer: int,
        source_activation: torch.Tensor,
        position: int = -1,
    ) -> torch.Tensor:
        block = self.blocks[layer]

        def patch_hook(_module, _inputs, output):
            hidden = self._hidden(output)
            replacement = source_activation.to(device=hidden.device, dtype=hidden.dtype)
            if replacement.ndim == 3:
                replacement = replacement[:, position, :]
            elif replacement.ndim != 2:
                raise ValueError("source_activation must be [B,T,D] or [B,D]")
            patched = hidden.clone()
            patched[:, position, :] = replacement
            return self._replace_hidden(output, patched)

        handle = block.register_forward_hook(patch_hook)
        batch = self.tokenize(prompt)
        try:
            with torch.inference_mode():
                return self.model(**batch).logits
        finally:
            handle.remove()

    def _steering_hook(self, direction: torch.Tensor, coefficient: float, position: int | slice):
        def steering_hook(_module, _inputs, output):
            hidden = self._hidden(output)
            vector = direction.to(device=hidden.device, dtype=hidden.dtype)
            if vector.ndim != 1 or vector.shape[0] != hidden.shape[-1]:
                raise ValueError("direction must be a one-dimensional vector matching hidden size")
            steered = hidden.clone()
            steered[:, position, :] = steered[:, position, :] + coefficient * vector
            return self._replace_hidden(output, steered)

        return steering_hook

    def forward_with_steering(
        self,
        prompt: str,
        *,
        layer: int,
        direction: torch.Tensor,
        coefficient: float,
        position: int | slice = -1,
    ) -> torch.Tensor:
        handle = self.blocks[layer].register_forward_hook(self._steering_hook(direction, coefficient, position))
        batch = self.tokenize(prompt)
        try:
            with torch.inference_mode():
                return self.model(**batch).logits
        finally:
            handle.remove()

    def generate(self, prompt: str, *, max_new_tokens: int = 96, **generation_kwargs) -> str:
        """Generate text without intervention."""
        batch = self.tokenize(prompt)
        prompt_length = batch["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = self.model.generate(**batch, max_new_tokens=max_new_tokens, **generation_kwargs)
        generated_ids = output_ids[0, prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_with_steering(
        self,
        prompt: str,
        *,
        layer: int,
        direction: torch.Tensor,
        coefficient: float,
        position: int | slice = -1,
        max_new_tokens: int = 96,
        **generation_kwargs,
    ) -> str:
        """Generate while applying the same residual intervention at each decode step."""
        handle = self.blocks[layer].register_forward_hook(self._steering_hook(direction, coefficient, position))
        batch = self.tokenize(prompt)
        prompt_length = batch["input_ids"].shape[-1]
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**batch, max_new_tokens=max_new_tokens, **generation_kwargs)
        finally:
            handle.remove()
        generated_ids = output_ids[0, prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
