<h1 align="center">DDTree + Tree Exit Verification</h1>

<p align="center">
  Inference and evaluation code for <strong>Tree Exit Verification (TEV)</strong>,
  paired with DFlash draft models and DDTree construction.
</p>

[Back to the project README](../README.md) · [Gemma 4 guide](./docs/GEMMA4_DDTREE.md)

## Overview

This directory contains the end-to-end decoding path used by TEV:

1. a DFlash draft model predicts position-wise token distributions;
2. the DDTree builder selects a bounded prefix tree;
3. one target-model forward pass evaluates the tree;
4. TEV returns an accepted path and one bonus token;
5. the target KV cache is compacted to the accepted branch.

At `temperature > 0`, the DDTree branch uses TEV. At `temperature = 0`, it uses the cheaper greedy tree-follow path, which is equivalent under argmax decoding.

## Released draft models

| Target model | Draft model |
|---|---|
| `Qwen/Qwen3-4B` | [`husj576/TEV-qwen3-4B`](https://huggingface.co/husj576/TEV-qwen3-4B) |
| `Qwen/Qwen3-8B` | [`husj576/TEV-qwen3-8B`](https://huggingface.co/husj576/TEV-qwen3-8B) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | [`husj576/TEV-qwen3-coder-30B-A3B`](https://huggingface.co/husj576/TEV-qwen3-coder-30B-A3B) |
| `google/gemma-4-12B-it` | [`husj576/TEV-gemma4-it-12B`](https://huggingface.co/husj576/TEV-gemma4-it-12B) |
| `meta-llama/Llama-3.1-8B-Instruct` | [`husj576/TEV-llama31-8B`](https://huggingface.co/husj576/TEV-llama31-8B) |

Use each draft checkpoint only with its corresponding target model.

## Directory layout

```text
ddtree/
├── benchmark.py                        # Benchmark CLI
├── ddtree.py                           # Tree construction, TEV, and generation
├── dflash.py                           # AR and block-parallel DFlash paths
├── distributed.py                      # torchrun helpers
├── model/
│   ├── dflash.py                       # DFlash draft-model implementation
│   ├── target_loader.py                # Text target and Gemma 4 loading
│   └── utils.py                        # Sampling and dataset utilities
├── application/
│   └── webui.py                        # Gradio demo
├── scripts/
│   ├── run_ddtree_tev_smoke.sh         # Single-GPU smoke test
│   └── run_ddtree_tev_multi_dataset.sh # Eight-GPU evaluation sweep
├── docs/
│   └── GEMMA4_DDTREE.md                # Gemma 4 text-only target guide
└── requirements.txt
```

## Installation

A CUDA-enabled PyTorch environment is required. The DFlash drafter uses FlashAttention.

```bash
conda create -n tev-infer python=3.10 -y
conda activate tev-infer
pip install -r requirements.txt
```

For Gemma 4, install a Transformers release that recognizes the unified architecture:

```bash
pip install -U "transformers>=5.11"
```

Access to some target checkpoints may require accepting their Hugging Face terms and logging in with `hf auth login`.

## Quick start

Run commands from this `ddtree/` directory.

### Qwen3-4B smoke test

```bash
export TARGET_MODEL=Qwen/Qwen3-4B
export DRAFT_MODEL=husj576/TEV-qwen3-4B
bash scripts/run_ddtree_tev_smoke.sh
```

Defaults:

- dataset: `gsm8k`;
- temperature: `1.0`;
- prompts: `20`;
- tree budget: `64`;
- maximum new tokens: `2048`;
- GPU: `0`.

Override them with `DATASET`, `TEMPERATURE`, `MAX_SAMPLES`, `TREE_BUDGET`, `MAX_NEW_TOKENS`, and `GPU_ID`.

The run writes:

- `runs/ddtree_tev_smoke/<tag>.pt`: raw per-prompt generation records;
- `logs/ddtree_tev_smoke/<tag>__gpu<id>.log`: benchmark output and errors.

### Other released checkpoints

Switch only the matching target and draft IDs. For example:

```bash
# Qwen3-8B
TARGET_MODEL=Qwen/Qwen3-8B \
DRAFT_MODEL=husj576/TEV-qwen3-8B \
bash scripts/run_ddtree_tev_smoke.sh

# LLaMA-3.1-8B-Instruct
TARGET_MODEL=meta-llama/Llama-3.1-8B-Instruct \
DRAFT_MODEL=husj576/TEV-llama31-8B \
bash scripts/run_ddtree_tev_smoke.sh

# Qwen3-Coder-30B-A3B-Instruct
TARGET_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
DRAFT_MODEL=husj576/TEV-qwen3-coder-30B-A3B \
bash scripts/run_ddtree_tev_smoke.sh
```

For Gemma 4, see [the dedicated setup and behavior notes](./docs/GEMMA4_DDTREE.md).

## Benchmark CLI

The shell scripts are wrappers around `benchmark.py`:

```bash
torchrun --nproc_per_node=1 --master_port=29700 benchmark.py \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path husj576/TEV-qwen3-4B \
  --dataset gsm8k \
  --max-samples 128 \
  --max-new-tokens 2048 \
  --temperature 1.0 \
  --tree-budget 64 \
  --save-path runs/qwen3_4b/gsm8k__t1_0__tb64.pt
```

Without `--skip-baseline-dflash`, the command runs three paths:

- autoregressive decoding (`baseline`);
- block-parallel DFlash (`dflash`);
- DDTree + TEV (`ddtree_tb<budget>`).

Add `--skip-baseline-dflash` to run only DDTree + TEV.

### Key arguments

| Argument | Description |
|---|---|
| `--model-name-or-path` | Target model path or Hugging Face repository ID. |
| `--draft-name-or-path` | Matching DFlash/ExitTrain draft checkpoint. |
| `--dataset` | `gsm8k`, `humaneval`, `math500`, `mt-bench`, `mbpp`, `alpaca`, `aime24`, `aime25`, `lbpp`, `livecodebench`, or `swe-bench`. |
| `--tree-budget` | One value or a comma-separated budget sweep, such as `32,64,128`. |
| `--temperature` | Positive values use TEV sampling; `0` uses greedy tree-follow. |
| `--max-samples` | Optional limit on the number of benchmark examples. |
| `--max-new-tokens` | Maximum generated tokens per response. |
| `--skip-baseline-dflash` | Skip AR and block-parallel DFlash references. |
| `--disable-cpp-compact-cache` | Use Python KV-cache compaction instead of the inline C++ extension. |
| `--save-path` | Destination for the raw `.pt` trace. |

Do not pass `--flash-attn` when running DDTree: its custom tree mask requires the target SDPA path, and enabling target FlashAttention disables DDTree variants. The draft model still uses FlashAttention.

### Saved output

The saved `.pt` object contains:

- benchmark arguments and attention backends;
- generated token IDs;
- input and output token counts;
- time to first token and time per output token;
- output-block lengths for each decode round;
- decode-round counts and stage timings.

## Multi-dataset sweep

```bash
export TARGET_MODEL=Qwen/Qwen3-4B
export DRAFT_MODEL=husj576/TEV-qwen3-4B
export TREE_BUDGETS="32 64 128"
export REPS=3
bash scripts/run_ddtree_tev_multi_dataset.sh
```

The script launches eight workers over GSM8K, HumanEval, MT-Bench, and MATH-500 at temperatures `0.6` and `1.0`. It assumes an eight-GPU node; edit `JOBS=(...)` in the script for a different layout.

Outputs are stored under:

- `runs/ddtree_tev_multi_dataset/`;
- `logs/ddtree_tev_multi_dataset/`.

## Interactive demo

The Gradio demo compares DDTree + TEV with autoregressive decoding on the same prompt.

```bash
python -m application.webui \
  --target Qwen/Qwen3-4B \
  --draft husj576/TEV-qwen3-4B \
  --cuda-visible-devices 0 \
  --server-port 7860
```

The interface reports decoding throughput and mean tokens produced per round. It can also highlight tokens emitted by each speculative round.

Important options:

| Option | Default | Description |
|---|---:|---|
| `--tree-budget` | `64` | Initial DDTree budget; adjustable in the UI. |
| `--max-new-tokens` | `1024` | Maximum response length; adjustable in the UI. |
| `--server-name` | `0.0.0.0` | Bind address. |
| `--server-port` | `7860` | HTTP port. |
| `--share` | off | Enable a temporary Gradio share URL. |
| `--disable-cpp-compact-cache` | off | Disable the inline C++ KV-cache compactor. |

## Programmatic use

Use `ddtree_generate` for end-to-end generation. For verifier-only tests, `tev_verify` wraps the three TEV stages:

```python
from ddtree import tev_verify

path, path_tensor, bonus_token = tev_verify(
    logits=target_logits,
    verify_input_ids=verify_ids,
    parents=parents,
    parents_np=parents_np,
    child_maps=child_maps,
    temperature=temperature,
    path_buffer=path_buffer,
)
```

Latency-sensitive integrations should call `tev_prepare`, `tev_finalize`, and `tev_sample_bonus` separately, as done in `ddtree_generate`.

## Troubleshooting

- **`ModuleNotFoundError` for `ddtree`, `dflash`, or `model`:** run the command from the `ddtree/` directory, or add that directory to `PYTHONPATH`.
- **`flash_attn must be installed`:** install FlashAttention in the active environment; the draft model requires it.
- **C++ extension compilation fails:** rerun with `--disable-cpp-compact-cache`.
- **Startup is slow:** the inline cache-compaction extension is compiled when the benchmark or demo enables it and is cached under `~/.cache/torch_extensions/`.
- **Gemma 4 config is not recognized:** upgrade to `transformers>=5.11` and follow [`docs/GEMMA4_DDTREE.md`](./docs/GEMMA4_DDTREE.md).
- **Out of memory:** reduce `TREE_BUDGET` or `MAX_NEW_TOKENS`; Gemma 4 and Qwen3-Coder require substantially more GPU memory than Qwen3-4B/8B.
