# Fast-scGPT

Reference implementation of an scGPT-style single-cell transformer with a few **modded-nanogpt**-style tweaks. Training is built around **[SLAF](https://github.com/slaf-project/slaf)** datasets and dataloaders (including the queue-based **distributed dataloader** on multi-GPU Modal) and **[Modal](https://modal.com/)** for cloud GPUs. Datasets and org assets on Hugging Face: **[slaf-project](https://huggingface.co/slaf-project)**.

## What you get

- **`fast_scgpt/`** — model (`ScGPT`), configs, single-process training (`train.py`), and native DDP entrypoint (`train_ddp.py`) used under `torchrun` on Modal.
- **`modal_train.py`** — one GPU on Modal (good for smoke tests and throughput on a single device).
- **`modal_train_distributed.py`** — 8× GPU on one Modal node, or 16× GPU across two nodes (Modal multi-node), using SLAF’s distributed pipeline via `slafdb`.

## Prerequisites

- **Python 3.11+**
- A **[Modal](https://modal.com/)** account and the CLI configured on your machine. Modal’s own docs are the best place to start: **[Modal documentation](https://modal.com/docs)** (install CLI, `modal token set`, workspaces, secrets, volumes).
- A Modal **Volume** named `slaf-datasets` (or change the volume name in the Modal scripts to match yours). Benchmark JSON is written under `/data/benchmark_results/` on that volume.
- **Data**: training reads SLAF-style data. For a minimal path that does not require your own S3 bucket, use **`--data-source hf`**, which points at the public Hugging Face dataset layout
  [`slaf-project/Tahoe-100M`](https://huggingface.co/datasets/slaf-project/Tahoe-100M) (see Modal script logic for the exact `hf://` path).
- **`--data-source s3`**: expects a Modal secret named **`s3-credentials`** and the default bucket layout used in the scripts (adjust if your infra differs).
- **`--data-source volume`**: expects data at `/data/tigris/...` on the mounted volume (see `modal_train.py`).

### Clone and environment

```bash
git clone https://github.com/slaf-project/fast-scgpt.git
cd fast-scgpt
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev,modal]"
```

The editable install pulls in PyTorch, dev tools, and Modal so you can run `modal run ...` from the repo root. It is **not** implying a stable package on PyPI.

### Distributed Modal runs (local machine)

For `modal run modal_train_distributed.py`, the local entrypoint calls **`deploy_dataloader_app`** from `slaf`. You need the ML extra installed **locally** (not only inside the Modal image), for example:

```bash
uv pip install "slafdb[ml]"
```

If that import fails, follow the [slaf](https://github.com/slaf-project/slaf) repo for an up-to-date install.

## Quickstart (Modal, small model, fixed steps)

All commands run from the **repository root**. Flags use Modal’s CLI style (`--data-source`, `--model-size`, …).

Use a **small** model and **short** run first; scale `n_steps`, `max_genes`, and `batch_size` once things work.

### Single GPU (1× H100)

```bash
modal run modal_train.py \
  --model-size small \
  --n-steps 100 \
  --batch-size 8 \
  --max-genes 128 \
  --data-source hf
```

### Single node, 8× GPU (DDP + SLAF queue dataloader)

```bash
modal run modal_train_distributed.py \
  --model-size small \
  --n-steps 100 \
  --batch-size 8 \
  --max-genes 128 \
  --data-source hf
```

### Multi-node, 16× GPU (2× 8× H100, Modal clustered / beta)

```bash
modal run modal_train_distributed.py \
  --multinode \
  --model-size small \
  --n-steps 100 \
  --batch-size 8 \
  --max-genes 128 \
  --data-source hf
```

Multi-node uses Modal’s experimental clustered API (`efa_enabled` in the image); treat it as **beta** and confirm against [Modal’s multi-node guidance](https://modal.com/docs).

## Alternate configs (cheat sheet)

| Flag / topic | Notes |
|--------------|--------|
| **`model-size`** | Presets: `small` **35,362,304**, `scgpt` **51,061,760**, `base` **102,045,696**, `large` **430,632,960** trainable parameters (default `ModelConfig` vocab / bins; **`scgpt`** uses weight tying). Matches `fast_scgpt.training_metrics.get_param_count()`. **Modal** `modal_train.py` / `modal_train_distributed.py`: all four. **Local** `python -m fast_scgpt.train`: **`--model_size`** is only **`small` \| `base` \| `large`** today—no CLI preset for `scgpt` (use Python / Modal for `scgpt`). |
| **`data-source`** | `hf`, `s3`, `volume` (see Prerequisites). Default **both** Modal scripts: **`s3`**. |
| **`modal_train.py`** | Extra knobs not present on distributed: `--compile-mode`, `--use-swiglu`, `--use-lp-layernorm`, `--use-softcap`, `--use-strict-bf16`, `--torch-profiler-steps`, `--torch-profiler-warmup-steps`, `--torch-profiler-chrome-path`, etc. Shared: `--use-compile`, `--use-gradient-checkpointing`, `--sparse-gene-head`, `--profile`, **`--flash-attn-backend`** (`fa3` default, **`fa4`** for the Flash Attention 4 image). See `main()` in `modal_train.py`. |
| **`modal_train_distributed.py`** | Per-GPU batch = `batch_size`; effective global batch ≈ **`batch_size × gradient_accumulation_steps × num_gpus`** (8 single-node, 16 with `--multinode`). Flags include **`--use-compile`**, **`--sparse-gene-head`**, **`--profile`**, **`--multinode`**, **`--flash-attn-backend`**. No SwiGLU / LP-LayerNorm / softcap / strict-bf16 toggles on this entrypoint—use `modal_train.py` or change the script if you need them. |
| **Local CPU/GPU (no Modal)** | `python -m fast_scgpt.train --slaf_path /path/to/dataset.slaf` (install **`slaf`** / dataset I/O deps as needed). Defaults in argparse: `batch_size=32`, **`max_genes=512`**, `log_every=10`, plus **torch-profiler** flags; see `main()` in `fast_scgpt/train.py`. |

**Modal defaults (when you pass no overrides):** `modal_train.py` → `batch_size=32`, `max_genes=64`, **`model_size=small`**, `data_source=s3`. `modal_train_distributed.py` → **`batch_size=64`**, **`max_genes=1024`**, **`model_size=base`**, `data_source=s3`. Match these explicitly when comparing scripts.

## Benchmarks

### `scgpt` (51,061,760 parameters) on 8× NVIDIA H100 (single node, 80GB per GPU)

Representative **Modal** distributed run via `modal_train_distributed.py`:

- **Attention:** Flash Attention 4
- **Run length:** 50 steps (short benchmark; see note below on longer jobs)
- **Batch:** 128 cells per GPU per step → **effective global batch 1024**
- **Sequence:** max genes **512**
- **Data:** Tahoe-100M, streamed into Modal from **S3** through the distributed dataloader with **2 CPU** prefetch workers

**Command** (from repo root; requires Modal + `s3-credentials` and the SLAF distributed dataloader deploy as in `modal_train_distributed.py`):

```bash
modal run modal_train_distributed.py \
  --model-size scgpt \
  --n-steps 50 \
  --batch-size 128 \
  --max-genes 512 \
  --data-source s3 \
  --flash-attn-backend fa4 \
  --sparse-gene-head \
  --no-use-compile
```

| Metric | Value |
|--------|--------|
| **Median step time** | **56 ms** (steady state; sub-100 ms per step at 128 cells/GPU) |
| **Global throughput** | **~18.3k cells/s** |
| **Steps/s** | **~18** |
| **Training compute (50 steps)** | **~23.5 s** wall time (end-to-end job ~95 s including startup/teardown) |
| **Peak GPU memory** | **~71.7 GB / GPU** (~84%) |
| **MFU** | **~17.3%** |
| **Achieved TFLOPS** | **~1370 total** (~171 per GPU) |

**How these numbers are defined (8-GPU run)** — implementation detail in `fast_scgpt/train_ddp.py`:

- **Median step time** and **peak GPU memory:** each step uses a **max reduce over the 8 ranks** (slowest rank’s end-to-end step time; highest peak VRAM among peers). The printed **median** is over those per-step maxima (warmup step excluded when present).
- **`nvidia-smi` GPU utilization** and **`dmon` SM efficiency:** **sampled on rank 0 only** (avoids an all-gather of NVML samples every step).
- **MFU:** **computed on rank 0** from estimated model FLOPs and global throughput (throughput already reflects the cross-rank **max** step time); it is not an average of per-GPU MFU.

**Caveats:**

- For judging this stack, treat **step time**, **cells/s**, and **MFU** as the main signal—they already encode steady-state training cost once the first steps settle.
- On a longer run, one-time costs (graph/JIT compile, allocator warmup, loader ramp-up) amortize: wall time per step trends toward the steady median you see after warmup, and the job-level time dominated by startup/teardown (Modal, queue workers, process group setup) shrinks as a fraction of total time.
- Don't expect `nvidia-smi` util or rank-0 `dmon` SM% to become perfect proxies for “cluster efficiency”. They still mix **idle gaps between kernels** (Python, NCCL, sync) with compute on one GPU, so use them alongside step-time/MFU, or profile with a proper tool if you need duty-cycle truth on every device.

### `scgpt` (51,061,760 parameters) on 1× NVIDIA H200 (Modal)

Representative **`modal_train.py`** run (single-process `fast_scgpt.train`). **Flash Attention 4**, **`--no-use-compile`**, **`--sparse-gene-head`**, **`--profile`** (step timing breakdown), **50 steps**, **128 cells/step** (effective batch **128**), **max genes 512**, **S3** data source.

**Command** (from repo root):

```bash
modal run modal_train.py \
  --model-size scgpt \
  --n-steps 50 \
  --batch-size 128 \
  --max-genes 512 \
  --data-source s3 \
  --flash-attn-backend fa4 \
  --profile \
  --sparse-gene-head \
  --no-use-compile
```

**Breakdown of time within training step (steady state)** — With **`profile=True`**, logs include chunks measured inside `train_step()` (CUDA-synchronized intervals; see `fast_scgpt/train.py`). Below is a representative step.

| Phase | Time (ms) | What it measures |
|-------|-----------|------------------|
| **`dl`** | **0** | Host time blocked on `next(batch_iter)` until the batch is produced. **~0 ms** here means the iterator returns immediately: **prefetch / overlap** is feeding the GPU without stalling this timer. |
| **`data`** | **1** | `input_ids` / `attention_mask` **host to device** copy. |
| **`mask`** | **1** | **Masked-language-model masking setup** (`create_mask`: which gene/expression tokens to predict and the corresponding targets/masks). |
| **`fwd`** | **112** | **Forward + loss** (`model.compute_loss` under autocast). |
| **`bwd`** | **202** | **Backward** (`loss.backward`). |
| **`opt`** | **3** | **Optimizer** (`step`, `zero_grad`; scaler update when AMP is on). |

| Metric | Value |
|--------|--------|
| **Median step time** | **323.3 ms** (summary excludes first-batch warmup) |
| **Avg step time** | **344.2 ms** |
| **Training throughput** | **396 cells/s** |
| **Steps/s** | **3.09** |
| **Peak GPU memory** | **67.92 GB** (**45%** of GPU in this allocation) |
| **MFU** | **23.71%** |
| **Achieved TFLOPS** | **74.0** (single device) |
| **`nvidia-smi` GPU util** | **84.9%** |
| **SM efficiency (`dmon`)** | **45.8%** |


**Util / SM%:** On this single-GPU trace, **`nvidia-smi` / `dmon`** sample the only training device for the whole job—usually easier to interpret than the short 8-GPU run’s rank-0-only hardware sample.
