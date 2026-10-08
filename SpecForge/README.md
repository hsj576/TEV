# ExitTrain: Exit-Guided Draft-Tree Training

This repository is an **[SpecForge](https://github.com/sgl-project/SpecForge)
fork** that adds a new training method called **ExitTrain**
(Exit-Guided Draft-Tree Training) — a speculative-decoding
draft-training objective that uses target-side **exit weights** on the
realized inference-time draft tree as node-level supervision.

ExitTrain is the training component of our **ExitTrain + TEV** system.
The verifier-side component (Tree Exit Verification, TEV) is
inference-only and lives outside this repository; the draft
checkpoints produced here are drop-in compatible with any standard
speculative-decoding verifier and, in particular, with TEV.

Everything else (data pipeline, disaggregated online capture with SGLang,
KV transport with Mooncake, distributed trainer, checkpoint export)
stays identical to upstream SpecForge and is fully compatible with the
`specforge train` / `specforge export` CLI.

> **License / provenance.** This fork is distributed under the same MIT
> license as upstream SpecForge (see [`LICENSE`](./LICENSE) and
> [`NOTICE`](./NOTICE)).


## Contents

- [1. What ExitTrain does](#1-what-exittrain-does)
- [2. Repository layout](#2-repository-layout)
- [3. Installation](#3-installation)
- [4. Data & model preparation](#4-data--model-preparation)
- [5. Running ExitTrain](#5-running-exittrain)
- [6. Monitoring, checkpoints, resuming](#6-monitoring-checkpoints-resuming)
- [7. Exporting to HuggingFace / SGLang](#7-exporting-to-huggingface--sglang)


## 1. What ExitTrain does

ExitTrain (`strategy: "exittrain"`) supervises a DFlash-family drafter
with **exit-guided** node-level rewards on the *inference-time* draft
tree. It uses the same tree builder that will be used at inference —
our default is DDTree's cumulative-score best-first builder with a
budget of `tree_budget` nodes — so training operates directly on the
finite-budget trees the drafter will actually produce at serving time.

For every training context, ExitTrain:

1. Builds the inference-time draft tree from the current drafter with
   the same tree builder used at inference (bounded to `tree_budget`
   nodes).
2. Evaluates every non-root node `v` with the frozen target model
   under its **actual tree-parent context** — i.e., the target sees
   `context | path_to_parent(v)` — using one target forward with a
   tree attention mask so all nodes are scored in a single pass.
3. Computes the target-side **exit weight** at each node:

   ```
   w(v) = P(v) * rho(v)
   ```

   where `P(v)` is the target-prefix probability of `v` and `rho(v) =
   1 - sum_{u in children(v)} p_v(x_u)` is the missing next-token mass
   below `v`. Intuitively, `w(v)` is the amount of one-step target
   coverage currently *missing* below `v`: prefixes the target
   frequently reaches but that the current tree fails to expand.
4. Uses `w(v)` to reweight the drafter's dense token loss on the
   incoming token of each non-root node, producing the exit-guided
   auxiliary loss:

   ```
   L_exit(θ) = -( Σ_v  w(v) * log q_{θ, depth(v)}(x_v | c) ) / Σ_v 1
   ```

   The dense draft distribution `q_{θ, depth(v)}` is DFlash's
   position-wise marginal at future draft position `depth(v)`; nodes
   at the same depth share the same draft row, while their target
   weights `w(v)` differ because `w(v)` is computed under each node's
   own tree-parent context. All target-derived quantities and the
   tree structure are detached; gradient flows only into the drafter.
5. Optimizes a hybrid objective

   ```
   L(θ) = L_tok(θ) + λ_exit * L_exit(θ)
   ```

   where `L_tok` is the standard dense token-prediction loss ("what to
   predict") and `L_exit` is the tree-allocation surrogate ("where the
   bounded tree should spend its nodes"). Increasing the marginal
   probability of `x_v` at position `depth(v)` raises the future
   priority of `v` under the cumulative-score builder, so the builder
   is more likely to retain and further expand exactly those regions
   where target mass is currently leaving the tree.

Because `L_exit` operates directly on the inference-time draft tree
and localizes uncovered target mass at the node level, it differs
from token- or feature-level alignment, position-wise distillation,
and scalar tree-reward optimization: those either ignore the tree
structure or provide only path-level scalar rewards, whereas ExitTrain
provides node-level supervision that says *which* branches deserve
more of the finite tree budget.


## 2. Repository layout

The **new** ExitTrain code lives under a single directory:

```
specforge/algorithms/exittrain/
├── __init__.py            # exports create_registration()
├── providers.py           # registers strategy="exittrain"; warm-start / capture
├── model.py               # OnlineExitTrainModel (subclass of OnlineDFlashModel)
├── loss.py                # exit-weight tensor build + L_exit reduction
└── tree_attention.py      # frozen target-model verifier + tree attention mask
```

Reused verbatim from upstream SpecForge:

- `specforge/algorithms/common/`       — DFlash-family model base, dataset providers
- `specforge/application/`             — run assembly (offline/online, colocated/disaggregated)
- `specforge/runtime/`                 — control / data plane (Mooncake ingest, feature store)
- `specforge/training/`                — trainer loop, FSDP backend, checkpoint I/O
- `specforge/modeling/draft/dflash*`   — draft network architectures
- `specforge/export/`                  — HuggingFace / SGLang exporter
- `specforge/cli.py`                   — the `specforge train` / `export` entry point

ExitTrain-specific configs & launch scripts:

```
examples/configs/online/disaggregated/managed-local/
├── qwen3-4b-exittrain.yaml
├── qwen3-8b-exittrain.yaml
├── llama3.1-8b-exittrain.yaml
└── qwen3-coder-30b-a3b-exittrain.yaml

run_qwen3_4b_exittrain.sh
run_qwen3_8b_exittrain.sh
run_llama31_8b_exittrain.sh
run_qwen3_coder_30b_a3b_exittrain.sh
```

Unit test:

```
tests/test_utils/test_exittrain_tree_attention.py    # CPU-only, no target-model download
```


## 3. Installation

ExitTrain requires **Python 3.10+**, a CUDA-capable GPU (H20 / H100 / A100
class, ≥80 GB HBM per rank for the 8B / 30B recipes), and the same
software stack as upstream SpecForge.

### 3.1. System prerequisites

- NVIDIA driver + CUDA 12.x
- git, tmux/screen (recommended for long-running training)
- 100+ GB free disk for weight downloads + capture cache + checkpoints

### 3.2. Create a fresh conda environment

```bash
conda create -n specforge python=3.10 -y
conda activate specforge
```

### 3.3. Install this fork

```bash
git clone <this-repo-url>
cd SpecForge
pip install -e .
```

`pip install -e .` reads [`pyproject.toml`](./pyproject.toml) and pulls the
required stack:

| Package | Version | Purpose |
| --- | --- | --- |
| `torch` | `2.11.0` | Training / target-verify forward |
| `transformers` | `5.8.1` | Target model loader + tokenizer |
| `sglang` | `0.5.14` | Online capture server (`target_backend: "sglang"`) |
| `accelerate` | latest | FSDP wrapper |
| `pydantic` | latest | Typed config validation |
| `datasets`, `safetensors`, `huggingface-hub` | latest | Data / weight I/O |
| `yunchang` | latest | USP sequence-parallel primitives |
| `wandb`, `tensorboard` | latest | Metric logging (offline by default) |

After install, verify the CLI:

```bash
specforge train --help
```

### 3.4. Optional extras

Recommended for full performance:

```bash
pip install -e '.[fa]'      # flash-attn (H100/A100 kernels)
pip install -e '.[liger]'   # liger-kernel fused ops
pip install swanlab         # if you want SwanLab tracking
```

### 3.5. Mooncake KV transport

The ExitTrain recipes ship with `deployment.mode: disaggregated` and
`backend: mooncake` — the SGLang capture server writes target hidden
states into a shared Mooncake segment that the trainer consumes.
Mooncake is installed with SGLang and needs a machine-wide RDMA / TCP
segment; the shipped YAMLs already set `protocol: tcp` for portability.
No extra install step is required beyond `pip install sglang==0.5.14`.


## 4. Data & model preparation

### 4.1. Target model

Download the HuggingFace directory for whichever target you want to train
against, e.g.:

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir <PATH_TO_TARGET_MODEL>/Qwen3-4B
```

Supported target choices in this fork:

| Recipe | Target model | HF repo |
| --- | --- | --- |
| `qwen3-4b-exittrain.yaml` | Qwen3-4B | `Qwen/Qwen3-4B` |
| `qwen3-8b-exittrain.yaml` | Qwen3-8B | `Qwen/Qwen3-8B` |
| `llama3.1-8b-exittrain.yaml` | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen3-coder-30b-a3b-exittrain.yaml` | Qwen3-Coder-30B-A3B-Instruct | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |

### 4.2. Warm-start draft checkpoint

ExitTrain **requires a warm-started draft**. Any DFlash-family checkpoint
(`dflash` / `dted` / `exittrain`) trained on the same target is valid; we
recommend the DFlash-b16 checkpoint as the initial warm-start when
available. Point `model.draft_checkpoint_path` in the YAML at that
directory.

If you don't yet have a warm-start, you can produce one by first running
DFlash training on the same target with the corresponding
`examples/configs/.../qwen3-*b-dflash-*.yaml` recipe. See the upstream
[`docs/basic_usage/training.md`](./docs/basic_usage/training.md).

### 4.3. Training data

The recipes expect a **JSONL of chat conversations** at
`data.train_data_path` (one JSON dict per line with a
`conversations: [{role, content}, ...]` field). The `chat_template` field
in the YAML selects tokenization:

| Chat template | Used by |
| --- | --- |
| `"qwen"` | Qwen3-4B / Qwen3-8B / Qwen3-Coder-30B |
| `"llama3"` | Llama-3.1-8B-Instruct |

Public datasets that work out-of-the-box (roughly analogous to what we
used internally):

- [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- [`teknium/OpenHermes-2.5`](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

If your JSONL uses a different field layout you can either preprocess it
into the SpecForge format, or write a small custom loader plugged in via
the SpecForge data providers. See
[`scripts/prepare_data.py`](./scripts/prepare_data.py) for reference
preprocessing.


## 5. Running ExitTrain

Every ExitTrain run goes through the standard SpecForge CLI:

```bash
specforge train --config <ExitTrain YAML>
```

### 5.1. Configure the YAML

Open the recipe you want to run, e.g.
[`examples/configs/online/disaggregated/managed-local/qwen3-4b-exittrain.yaml`](./examples/configs/online/disaggregated/managed-local/qwen3-4b-exittrain.yaml),
and replace the placeholders with your local paths:

| Placeholder | Meaning |
| --- | --- |
| `<PATH_TO_TARGET_MODEL>/Qwen3-4B` | HF directory of the target model |
| `<PATH_TO_DRAFT_CHECKPOINT>/Qwen3-4B-DFlash-b16` | Warm-start DFlash-family checkpoint |
| `<PATH_TO_TRAIN_DATA>/.../perfectblend_train_regen.jsonl` | Training JSONL |
| `<PATH_TO_CACHE>/qwen3-4b-exittrain` | Cache dir for the tokenized dataset |
| `<PATH_TO_OUTPUT>/qwen3-4b-exittrain` | Output directory for checkpoints + logs |

You can leave the rest of the YAML at its defaults; the shipped recipe
already has `num_anchors=32`, `tree_budget=32`, `num_epochs=2`,
`lr=1e-4`, `exittrain_prefix_window=64`.

### 5.2. Launch — option A: use the provided shell script

The four ExitTrain shell scripts self-locate `REPO_ROOT` from the script
path, so you can run them from anywhere:

```bash
# Qwen3-4B
./run_qwen3_4b_exittrain.sh

# Qwen3-8B
./run_qwen3_8b_exittrain.sh

# Llama-3.1-8B-Instruct
./run_llama31_8b_exittrain.sh

# Qwen3-Coder-30B-A3B-Instruct (MoE, needs ≥143 GB HBM)
./run_qwen3_coder_30b_a3b_exittrain.sh
```

Each script:

1. Sources your conda env if `CONDA_HOME` / `CONDA_ENV` are set.
2. Puts `WANDB_MODE=offline` and lets you override `SWANLAB_API_KEY` via
   the environment.
3. Runs `nohup setsid specforge train ...` in the background,
   writing logs to `outputs/<recipe>/train.log` and its PID to
   `outputs/<recipe>/train.pid`.

Common overrides:

```bash
# Custom conda location & env name
CONDA_HOME=/opt/conda CONDA_ENV=specforge ./run_qwen3_4b_exittrain.sh

# Custom output directory
OUT_DIR=/data/runs/my-qwen3-4b-run ./run_qwen3_4b_exittrain.sh

# Enable SwanLab sync
SWANLAB_API_KEY=<your-key> ./run_qwen3_4b_exittrain.sh
```

### 5.3. Launch — option B: call the CLI directly

If you prefer manual control (e.g. under `tmux`):

```bash
conda activate specforge
cd /path/to/SpecForge
export WANDB_MODE=offline
specforge train --config examples/configs/online/disaggregated/managed-local/qwen3-4b-exittrain.yaml
```

You can override any config field on the command line:

```bash
specforge train \
  --config examples/configs/online/disaggregated/managed-local/qwen3-4b-exittrain.yaml \
  training.num_epochs=3 \
  training.learning_rate=5e-5 \
  output_dir=/data/runs/qwen3-4b-lr5e5
```

### 5.4. GPU assignment

The shipped recipes are for a **single 8-GPU node**:

- **GPU 0** hosts the SGLang capture server (runs the target model).
- **GPUs 1–7** run the trainer (DP=7, FSDP shards the draft).

If your node has fewer / more GPUs, edit the `deployment.disaggregated.managed_local`
section in the YAML:

- `trainer_cuda_visible_devices`: list of GPU IDs the trainer uses.
- `capture_servers[*].cuda_visible_devices`: GPU IDs for SGLang.
- `deployment.trainer.nproc_per_node`: must equal
  `len(trainer_cuda_visible_devices)`.

For example, on a 4-GPU node:

```yaml
deployment:
  trainer:
    nproc_per_node: 3
  disaggregated:
    managed_local:
      trainer_cuda_visible_devices: ["1", "2", "3"]
      capture_servers:
        - port: 30000
          cuda_visible_devices: ["0"]
          tp_size: 1
          mem_fraction_static: 0.5
```

### 5.5. Dry-run the launch plan

To print the resolved process plan without actually starting workers:

```bash
specforge train --config <yaml> --plan
```


## 6. Monitoring, checkpoints, resuming

### 6.1. Live log

```bash
tail -f <OUT_DIR>/train.log
```

### 6.2. Metrics

Metrics are streamed to WandB (offline by default; upload later with
`wandb sync`) and optionally to SwanLab if `SWANLAB_API_KEY` is set. Key
ExitTrain scalars:

- `train/loss_tok` — dense token-prediction loss `L_tok` (main term)
- `train/loss_exit` — exit-guided auxiliary loss `L_exit`
- `train/w_mean` — mean per-node exit weight `w(v)` across the batch
- `train/expected_al_per_anchor` — expected accepted length per anchor
- `train/grad_norm`, `train/lr`

Note: some legacy scalars in the tracker still use the DFlash-family
internal names (e.g. `loss_ce` for `L_tok`, `loss_dted` for `L_exit`).
They refer to exactly the same quantities as the paper's `L_tok` and
`L_exit`.

### 6.3. Checkpoints

The trainer writes checkpoints to `<OUT_DIR>/checkpoint-<STEP>/`
every `training.save_interval` steps (default `5000`). Each contains:

- FSDP-sharded draft weights (`state_dict/`)
- Optimizer / lr-scheduler state
- The resolved run config + provenance sidecar

### 6.4. Resume

Add `training.resume_from: "<OUT_DIR>/checkpoint-<STEP>"` (or a run root
URI) to the YAML — or pass it on the CLI:

```bash
specforge train --config <yaml> \
  training.resume_from=<OUT_DIR>/checkpoint-30000
```

### 6.5. Stopping a run

```bash
kill -TERM $(cat <OUT_DIR>/train.pid)
```

`SIGTERM` is caught by the CLI (see `_worker_signal_unwind` in
[`specforge/cli.py`](./specforge/cli.py)) so the trainer flushes the
current checkpoint and cleanly shuts down the SGLang capture server
before exiting.


## 7. Exporting to HuggingFace / SGLang

Once a checkpoint is trained, materialize it as a serving-ready HF
directory using the standard SpecForge exporter:

```bash
specforge export --to hf \
  --checkpoint <OUT_DIR>/checkpoint-<STEP> \
  --draft-config configs/qwen3-4b-dted.json \
  --output-dir <HF_EXPORT_DIR>/qwen3-4b-exittrain-hf \
  --embedding-source <PATH_TO_TARGET_MODEL>/Qwen3-4B \
  --embedding-key model.embed_tokens.weight
```

The resulting directory can be served with SGLang's speculative-decoding
mode; refer to the upstream SGLang docs for the exact `sglang.launch_server`
flags.

For the raw SGLang layout instead of HF:

```bash
specforge export --to sglang \
  --checkpoint <OUT_DIR>/checkpoint-<STEP> \
  --draft-config configs/qwen3-4b-dted.json \
  --output-dir <SGLANG_EXPORT_DIR>/qwen3-4b-exittrain-sglang
```
