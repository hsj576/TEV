# Running DDTree + TEV on Gemma 4 12B

This guide explains how to run the text-only DDTree + TEV benchmark with:

- target: [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it);
- draft: [`husj576/TEV-gemma4-it-12B`](https://huggingface.co/husj576/TEV-gemma4-it-12B).

[Back to the DDTree README](../README.md) · [Back to the project README](../../README.md)

## Scope

Gemma 4 12B uses a unified multimodal top-level architecture and a hybrid sliding/full-attention language backbone. DDTree expects a causal-LM-like target with a full-length custom tree attention mask.

The compatibility code in [`model/target_loader.py`](../model/target_loader.py) bridges these interfaces. No Gemma-specific changes are required in `benchmark.py`, `ddtree.py`, or `dflash.py`.

This path is currently **text-only**: it uses the Gemma 4 language backbone and tokenizer, but does not process image or audio inputs.

## Requirements

Gemma 4 unified models require a recent Transformers release. Create or update the inference environment as follows:

```bash
conda create -n tev-gemma4 python=3.10 -y
conda activate tev-gemma4

cd /path/to/TEV/ddtree
pip install -r requirements.txt
pip install -U "transformers>=5.11"
```

The target checkpoint may require accepting Google's terms on Hugging Face. If needed:

```bash
hf auth login
```

A CUDA-enabled PyTorch installation and `flash-attn` are required. The target uses SDPA for DDTree's custom tree mask; the DFlash drafter uses FlashAttention.

## Quick start

Run the generic smoke-test launcher from `TEV/ddtree`:

```bash
export TARGET_MODEL=google/gemma-4-12B-it
export DRAFT_MODEL=husj576/TEV-gemma4-it-12B
export TREE_BUDGET=64
export MAX_SAMPLES=20
export TEMPERATURE=1.0

bash scripts/run_ddtree_tev_smoke.sh
```

The result is written to `runs/ddtree_tev_smoke/`, with logs under `logs/ddtree_tev_smoke/`.

## Direct benchmark command

```bash
torchrun --nproc_per_node=1 --master_port=29700 benchmark.py \
  --model-name-or-path google/gemma-4-12B-it \
  --draft-name-or-path husj576/TEV-gemma4-it-12B \
  --dataset gsm8k \
  --max-samples 128 \
  --max-new-tokens 2048 \
  --temperature 1.0 \
  --tree-budget 64 \
  --save-path runs/gemma4_12b/gsm8k__t1_0__tb64.pt
```

Omit `--skip-baseline-dflash` to run the autoregressive and block-parallel DFlash references in the same process. Add it when only DDTree + TEV is needed.

Do not pass `--flash-attn` to `benchmark.py`: enabling target FlashAttention disables the DDTree branch because the target must consume a custom 4D tree mask through SDPA.

## Interactive demo

```bash
python -m application.webui \
  --target google/gemma-4-12B-it \
  --draft husj576/TEV-gemma4-it-12B \
  --cuda-visible-devices 0 \
  --tree-budget 64 \
  --server-port 7860
```

The demo compares DDTree + TEV with autoregressive decoding and reports decoding speed and mean tokens produced per round.

## How the compatibility layer works

All Gemma 4 handling is localized to [`model/target_loader.py`](../model/target_loader.py).

### 1. Load the unified model and extract the language path

`load_target` detects that the top-level configuration contains a distinct `text_config`, then loads the target with `AutoModelForImageTextToText`. `TextOnlyCausalLMWrapper` exposes the language backbone and top-level LM head through the causal-LM interface expected by DDTree.

The wrapper provides:

- the text embedding module;
- the target LM head;
- the text configuration;
- causal-LM-style `forward` outputs containing logits, hidden states, and KV cache.

### 2. Make the target cache compatible with the tree mask

Gemma 4 interleaves sliding-attention and full-attention layers. A standard `DynamicCache` may trim sliding-layer KV states to the native window, while DDTree's custom tree mask spans the complete cached sequence.

The loader exposes a shallow, sanitized copy of the text configuration to DDTree. In this copy, per-layer attention types are treated as full attention and the sliding-window fields are disabled. The underlying model configuration and weights are not modified.

### 3. Disable runtime KV trimming

After loading the language backbone, `_disable_sliding_attention_runtime` disables sliding-window behavior on attention modules that expose `is_sliding` or `sliding_window`. This prevents the runtime from shortening K/V tensors relative to the full-length DDTree mask.

### 4. Preserve the custom mask

`TextOnlyCausalLMWrapper.forward` can map DDTree's 4D tree mask to both full- and sliding-attention mask entries expected by hybrid-attention model code. This prevents the target implementation from silently rebuilding a shorter sliding-window mask.

## Important behavior note

For compatibility with DDTree's full-length tree mask, this loader promotes Gemma 4's sliding-attention layers to full attention at runtime.

This has two consequences:

1. **Memory and runtime can increase**, especially for long contexts.
2. **The target forward differs from native Gemma 4 once the sequence exceeds its configured sliding window.** The benchmark compares AR, DFlash, and DDTree paths using the same loaded target, but results beyond the native window should not be described as native Gemma 4 decoding behavior.

The checkpoint weights remain unchanged. For short sequences that do not exceed the native sliding window, the receptive-field difference does not arise.

## Draft checkpoint details

The released draft checkpoint is configured for the 48-layer Gemma 4 12B text backbone:

| Field | Value |
|---|---|
| Draft repository | `husj576/TEV-gemma4-it-12B` |
| Architecture | `DFlashDraftModel` |
| Draft block size | `16` |
| Target layers | `48` |
| Target feature layers | `[1, 10, 19, 27, 36, 45]` |
| Hidden size | `3840` |
| Logit soft cap | `30.0` |

The draft is loaded through the same local `DFlashDraftModel.from_pretrained` path as the Qwen and LLaMA checkpoints.

## Troubleshooting

### `gemma4_unified` is not recognized

Upgrade Transformers in the active environment:

```bash
pip install -U "transformers>=5.11"
```

Then verify the target configuration can be loaded:

```bash
python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-12B-it")
print(type(cfg).__name__, cfg.model_type)
PY
```

### Hugging Face access is denied

Accept the target model's terms on its Hugging Face page, then run:

```bash
hf auth login
```

### Tree verification reports an attention-mask or KV-cache shape mismatch

Confirm that:

- the benchmark imports `load_target` from this repository;
- the loader reports that sliding attention was disabled;
- the target attention backend is `sdpa`;
- `--flash-attn` was not passed to `benchmark.py`.

### CUDA out of memory

Gemma 4 12B, its draft model, the full target KV cache, and the draft tree must fit on the same GPU in the current batch-one implementation. Start with a smaller workload:

```bash
TREE_BUDGET=32 MAX_SAMPLES=5 MAX_NEW_TOKENS=512 \
TARGET_MODEL=google/gemma-4-12B-it \
DRAFT_MODEL=husj576/TEV-gemma4-it-12B \
bash scripts/run_ddtree_tev_smoke.sh
```

### Startup is slow

The inline C++ KV-cache compactor is compiled when the benchmark or demo enables it. If the host lacks a compiler, add `--disable-cpp-compact-cache` to a direct `benchmark.py` invocation or to the Web UI command.
