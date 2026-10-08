"""Auto target loader for benchmark.py.

Handles both:
  1) Standard text-only CausalLM models (Qwen3, Llama, ...) via AutoModelForCausalLM.
  2) VLM top-level configs (Gemma4Unified, Qwen3.5MoE, ...) via AutoModelForImageTextToText,
     then extract the text-only submodel + lm_head into a lightweight CausalLM-like wrapper.

Returned object always exposes the interface expected by ddtree_generate / dflash_generate:
  - .device, .dtype, .config (text config), .generation_config
  - .model with .embed_tokens
  - .lm_head
  - .forward(input_ids, position_ids, attention_mask, past_key_values,
             use_cache, logits_to_keep, output_hidden_states, **kw)
    -> object with .logits, .hidden_states, .past_key_values
  - .eval(), .to(device) return self (idempotent)
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class TextOnlyCausalLMWrapper(nn.Module):
    """Wraps a VLM (Vision-Language Model) so it looks like a text-only CausalLM.

    Given a VLM ``full_vlm``, this wrapper finds the language backbone submodule
    (``.language_model`` under ``.model`` typically) and combines it with the
    top-level ``lm_head`` to provide a standard CausalLM forward interface.
    """

    def __init__(
        self,
        full_vlm: nn.Module,
        text_backbone: nn.Module,
        lm_head: nn.Module,
        text_config,
        generation_config=None,
    ):
        super().__init__()
        # Keep a strong reference to the full VLM to prevent GC while we are
        # holding submodule references; but do NOT register it as a Module
        # child (that would double-count parameters).
        self._full_vlm = full_vlm
        self.model = text_backbone
        self.lm_head = lm_head
        # Expose a *sanitised* config to downstream code (DDTree/DynamicCache):
        # any per-layer sliding_attention markers are rewritten to full_attention
        # and sliding_window is cleared. Prevents DynamicCache(config=...) from
        # allocating per-layer sliding buffers that get trimmed to a shorter
        # length than DDTree's 4D tree attention mask. This does NOT alter the
        # underlying model's real config (which stays untouched to preserve
        # per_layer_config homogeneity checks inside the modeling code).
        self.config = _sanitise_config_for_ddtree(text_config)
        self.generation_config = generation_config
        sc = getattr(self.config, "final_logit_softcapping", None)
        self._softcap = float(sc) if sc else None

    @property
    def device(self):
        return self.lm_head.weight.device

    @property
    def dtype(self):
        return self.lm_head.weight.dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def eval(self):
        self._full_vlm.eval()
        return self

    def to(self, *args, **kwargs):
        # Delegate to the underlying full VLM; return self for chaining.
        self._full_vlm.to(*args, **kwargs)
        return self

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=0,
        output_hidden_states=False,
        **kwargs,
    ):
        # Gemma4-family text backbones dispatch per-layer masks:
        #   causal_mask_mapping = { "full_attention": ..., "sliding_attention": ... }
        # picked by config.layer_types[i]. When DDTree passes a 4D tree mask
        # (Tensor), the model will call create_sliding_window_causal_mask() to
        # build a *shorter* sliding mask for sliding layers, producing a shape
        # mismatch with the K/V cache. We short-circuit that by wrapping the
        # tree mask into the mapping dict directly, so all layers use our full
        # tree mask uniformly (target_loader has already disabled `is_sliding`
        # on every attention module).
        if (
            attention_mask is not None
            and torch.is_tensor(attention_mask)
            and attention_mask.dim() == 4
            and getattr(self.config, "layer_types", None) is not None
            and "sliding_attention" in set(self.config.layer_types)
        ):
            attention_mask = {
                "full_attention": attention_mask,
                "sliding_attention": attention_mask,
            }

        lm_out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        h = lm_out.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            h_head = h[:, -logits_to_keep:, :]
        else:
            h_head = h
        logits = self.lm_head(h_head)
        if self._softcap is not None:
            logits = torch.tanh(logits / self._softcap) * self._softcap
        return SimpleNamespace(
            logits=logits,
            hidden_states=lm_out.hidden_states if output_hidden_states else None,
            past_key_values=lm_out.past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# Reflective helpers
# ---------------------------------------------------------------------------
def _find_text_backbone(full_vlm: nn.Module) -> Optional[nn.Module]:
    """Locate the text-only backbone submodule inside a VLM.

    Common patterns seen in transformers:
      - full.model.language_model   (Gemma4Unified, LLaVA, ...)
      - full.language_model         (some older VLMs)
      - full.model.text_model
      - full.text_model
    The backbone should expose ``.embed_tokens`` and ``.layers``.
    """
    candidates = []
    if hasattr(full_vlm, "model"):
        top = full_vlm.model
        for name in ("language_model", "text_model", "text_backbone"):
            if hasattr(top, name):
                candidates.append(getattr(top, name))
    for name in ("language_model", "text_model", "text_backbone"):
        if hasattr(full_vlm, name):
            candidates.append(getattr(full_vlm, name))
    for c in candidates:
        if hasattr(c, "embed_tokens") and hasattr(c, "layers"):
            return c
    return None


def _find_lm_head(full_vlm: nn.Module) -> Optional[nn.Module]:
    """Find the top-level lm_head (or output embeddings)."""
    if hasattr(full_vlm, "lm_head"):
        return full_vlm.lm_head
    if hasattr(full_vlm, "get_output_embeddings"):
        oe = full_vlm.get_output_embeddings()
        if oe is not None:
            return oe
    return None


def _text_config(full_vlm) -> object:
    cfg = full_vlm.config
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_target(
    model_path: str,
    attn_implementation: str = "sdpa",
    dtype=torch.bfloat16,
    device=None,
):
    """Load a target model from ``model_path``.

    Detection strategy:
      - If the config has a nested ``text_config`` with a different ``model_type``
        than the top-level (typical VLM pattern), load via AutoModelForImageTextToText
        and wrap the text backbone.
      - Otherwise, use AutoModelForCausalLM directly.
    Falls back gracefully if either path raises.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = {"attn_implementation": attn_implementation, "dtype": dtype}

    # First peek at the config to decide.
    cfg = AutoConfig.from_pretrained(model_path)
    is_vlm = False
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        top_mt = getattr(cfg, "model_type", None)
        text_mt = getattr(text_cfg, "model_type", None)
        # If the top-level model_type differs from the text sub-config's,
        # this is a wrapper (VLM) config. Even if AutoModelForCausalLM would
        # accept it, we want the wrapper so we operate purely on the text
        # backbone (avoids vision/audio kwargs contaminating forward calls).
        if top_mt and text_mt and top_mt != text_mt:
            is_vlm = True
        # Also treat as VLM if there is a vision_config or audio_config alongside.
        if hasattr(cfg, "vision_config") or hasattr(cfg, "audio_config"):
            is_vlm = True

    if not is_vlm:
        target = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        _disable_sliding_attention_runtime(target)
        if device is not None:
            target = target.to(device)
        target = target.eval()
        return target

    # VLM path: load full VLM, then wrap.
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as e:
        raise RuntimeError(
            f"Target {model_path} looks like a VLM but AutoModelForImageTextToText "
            f"is unavailable in this transformers version."
        ) from e
    target = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)

    backbone = _find_text_backbone(target)
    if backbone is None:
        raise RuntimeError(
            f"Loaded VLM {type(target).__name__} but could not locate a text backbone "
            f"submodule (looked for .model.language_model / .text_model / .text_backbone). "
            f"Top-level children: {[n for n, _ in target.named_children()]}"
        )
    lm_head = _find_lm_head(target)
    if lm_head is None:
        raise RuntimeError(
            f"Loaded VLM {type(target).__name__} but could not locate lm_head."
        )
    # Disable sliding attention on the text backbone before moving to device.
    _disable_sliding_attention_runtime(backbone)
    if device is not None:
        target = target.to(device)
    target.eval()
    wrapper = TextOnlyCausalLMWrapper(
        full_vlm=target,
        text_backbone=backbone,
        lm_head=lm_head,
        text_config=_text_config(target),
        generation_config=getattr(target, "generation_config", None),
    )
    return wrapper


def _sanitise_config_for_ddtree(text_config):
    """Return a *copy* of ``text_config`` with sliding attention neutralised.

    DDTree instantiates ``DynamicCache(config=target.config)``. For gemma4-family
    text configs this pre-allocates per-layer sliding buffers, which get trimmed
    to ``sliding_window`` positions during the SDPA attention forward. But
    DDTree's 4D tree attention mask always spans the full past length, so the
    trimmed K/V and the mask disagree in shape.

    We therefore expose a sanitised config on the wrapper: ``layer_types``
    becomes all ``full_attention`` and ``sliding_window`` is cleared. Only the
    *cache* (and downstream DDTree utilities that read ``config.layer_types``)
    see this modified copy; the underlying model's real config is untouched
    (which keeps per_layer_config homogeneity checks inside
    ``Gemma4UnifiedTextRotaryEmbedding`` happy).
    """
    import copy as _copy

    cfg = _copy.copy(text_config)
    try:
        if getattr(cfg, "layer_types", None) is not None:
            n = len(cfg.layer_types)
            cfg.layer_types = ["full_attention"] * n
    except Exception:
        pass
    for attr in ("sliding_window", "use_sliding_window"):
        if getattr(cfg, attr, None) not in (None, False):
            try:
                setattr(cfg, attr, None if attr == "sliding_window" else False)
            except Exception:
                pass
    return cfg


def _disable_sliding_attention_runtime(module) -> None:
    """Neutralise sliding-window attention on all leaf attention modules.

    DDTree passes a custom 4D tree attention mask that always spans the full
    past length. When the target model has sliding-attention layers (e.g.
    Gemma4Unified), the SDPA path will trim K/V to `sliding_window`, causing a
    shape mismatch with the mask. This helper walks the module tree and sets
    `is_sliding=False` / `sliding_window=None` on every attention module,
    which promotes those layers to full attention at runtime (weights are
    unchanged; only the KV-cache slicing behaviour is altered).

    We do NOT touch `config.layer_types` because gemma4-family models tie
    `per_layer_config` to the original layer types via a homogeneity check
    inside `Gemma4UnifiedTextRotaryEmbedding`.
    """
    import logging as _logging
    logger = _logging.getLogger(__name__)
    n_patched = 0
    for m in module.modules():
        # Match any attention submodule that has these runtime attributes.
        touched = False
        if getattr(m, "is_sliding", None):
            try:
                m.is_sliding = False
                touched = True
            except Exception:
                pass
        if getattr(m, "sliding_window", None) is not None:
            try:
                m.sliding_window = None
                touched = True
            except Exception:
                pass
        if touched:
            n_patched += 1
    if n_patched:
        logger.warning(
            "Disabled sliding attention on %d attention modules "
            "(promoted to full attention) so DDTree's 4D tree mask stays "
            "compatible with SDPA. Weights are unchanged.",
            n_patched,
        )
