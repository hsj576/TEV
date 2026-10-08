<h1 align="center">TEV: Exit-Guided Speculative Decoding</h1>

<p align="center">
  Official implementation of <strong>Where Draft Trees Lose Target Mass: Exit-Guided Speculative Decoding</strong>.
</p>

<p align="center">
  <strong>ExitTrain</strong> improves finite-budget draft trees, while
  <strong>Tree Exit Verification (TEV)</strong> executes the optimal fixed-tree verification law directly.
</p>

## Overview

Tree-based speculative decoding consists of a drafter, a bounded tree builder, and a target-model verifier. Our paper separates two complementary opportunities:

- **Better draft trees.** ExitTrain evaluates the realized inference-time tree with the frozen target model and uses node-level exit feedback to allocate the finite tree budget toward target-relevant prefixes.
- **More direct verification.** TEV realizes the canonical fixed-tree exit law with a regular verification procedure. It preserves the same fixed-tree acceptance law as saturated exact verifiers while reducing verifier-side execution cost.

```text
Training:  context -> DFlash drafter -> DDTree -> target tree evaluation -> ExitTrain
Inference: context -> trained drafter -> DDTree -> target tree forward -> TEV -> output block
```

Full definitions, derivations, and proofs are provided in the paper.

## Released draft models

The following ExitTrain checkpoints are available on Hugging Face. Each checkpoint is a **draft model** and must be paired with its corresponding target model.

| Target model | ExitTrain draft model | Draft block size |
|---|---|---:|
| [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | [`husj576/TEV-qwen3-4B`](https://huggingface.co/husj576/TEV-qwen3-4B) | 16 |
| [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | [`husj576/TEV-qwen3-8B`](https://huggingface.co/husj576/TEV-qwen3-8B) | 16 |
| [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | [`husj576/TEV-qwen3-coder-30B-A3B`](https://huggingface.co/husj576/TEV-qwen3-coder-30B-A3B) | 16 |
| [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | [`husj576/TEV-gemma4-it-12B`](https://huggingface.co/husj576/TEV-gemma4-it-12B) | 16 |
| [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [`husj576/TEV-llama31-8B`](https://huggingface.co/husj576/TEV-llama31-8B) | 10 |

## Quick start

The fastest way to run TEV is with a released draft checkpoint and the DDTree inference code.

```bash
git clone https://github.com/hsj576/TEV.git
cd TEV/ddtree

pip install -r requirements.txt

export TARGET_MODEL=Qwen/Qwen3-4B
export DRAFT_MODEL=husj576/TEV-qwen3-4B
bash scripts/run_ddtree_tev_smoke.sh
```

The smoke test runs GSM8K at temperature `1.0` with tree budget `64` by default. See [`ddtree/README.md`](./ddtree/README.md) for direct CLI usage, multi-dataset evaluation, the interactive demo, and configuration details.

For Gemma 4, first upgrade to a Transformers release that supports `Gemma4UnifiedForConditionalGeneration`, then follow the dedicated guide:

- [Running DDTree + TEV on Gemma 4](./ddtree/docs/GEMMA4_DDTREE.md)

## Repository layout

```text
TEV/
├── README.md
├── SpecForge/                         # ExitTrain training implementation
│   ├── specforge/algorithms/exittrain/
│   ├── examples/configs/              # Training recipes
│   └── README.md                      # Training and export instructions
└── ddtree/                            # DDTree construction and TEV inference
    ├── benchmark.py                   # Benchmark entry point
    ├── ddtree.py                      # DDTree + TEV generation loop
    ├── dflash.py                      # AR and DFlash reference paths
    ├── model/target_loader.py         # Text targets and Gemma 4 compatibility
    ├── application/webui.py           # Interactive demo
    ├── scripts/                       # Smoke test and evaluation launchers
    └── docs/GEMMA4_DDTREE.md          # Gemma 4 guide
```

## Training with ExitTrain

ExitTrain is implemented as a strategy in the included SpecForge fork. It fine-tunes a compatible DFlash-family checkpoint using target probabilities evaluated under the actual tree-parent contexts.

```bash
cd SpecForge
conda create -n tev-train python=3.10 -y
conda activate tev-train
pip install -e .
pip install -e '.[fa]'

specforge train \
  --config examples/configs/online/disaggregated/managed-local/qwen3-4b-exittrain.yaml \
  --plan

specforge train \
  --config examples/configs/online/disaggregated/managed-local/qwen3-4b-exittrain.yaml
```

Before launching, update the selected YAML with the target model, warm-start draft checkpoint, training data, cache, and output paths. The provided managed-local recipes assume one GPU for target rollout and seven GPUs for training.

See [`SpecForge/README.md`](./SpecForge/README.md) for data preparation, distributed launch, checkpoint management, and export instructions.

## Inference and evaluation

The `ddtree/` implementation supports:

- single-GPU smoke tests;
- direct benchmark runs on MT-Bench, HumanEval, GSM8K, MATH-500, and additional datasets;
- tree-budget sweeps;
- raw per-round acceptance and stage-timing traces;
- an interactive Gradio comparison between DDTree + TEV and autoregressive decoding;
- text-only inference with the Gemma 4 unified target through the compatibility path in `model/target_loader.py`.

Example direct invocation:

```bash
cd ddtree

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

At positive temperature, the DDTree branch uses TEV. At temperature `0`, it uses the cheaper greedy tree-follow path, which is equivalent under argmax decoding.

## Results reported in the paper

Across MT-Bench, HumanEval, GSM8K, and MATH-500, using the same DDTree builder and tree budget:

- ExitTrain + TEV improves average output-block length by up to **13%** over DDTree in the headline controlled comparison;
- the complete system improves end-to-end decoding throughput by up to **14%**;
- TEV reduces verifier-decision and cache-commit latency by approximately **15%** at tree budget `64`, while preserving the same saturated fixed-tree expected output-block length;
- the gains transfer across Qwen3, LLaMA 3.1, Gemma 4, and Qwen3-Coder target families.

The reported output-block length includes the bonus token produced in every speculative decoding cycle.

## Acknowledgements

This repository builds on [SpecForge](https://github.com/sgl-project/SpecForge), [DFlash](https://github.com/z-lab/dflash), and [DDTree](https://github.com/liranringel/ddtree). We thank the authors and maintainers for releasing their code and checkpoints.
