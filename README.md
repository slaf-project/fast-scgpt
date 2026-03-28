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
| **`model-size`** | `small` (dev), `scgpt` (~51M, paper-like), `base` (~102M), `large` (scaling). |
| **`data-source`** | `hf` (HF dataset URI), `s3` (needs `s3-credentials` secret), `volume` (path on Modal volume). |
| **`modal_train.py` only** | Optional Modal CLI flags (examples): `--use-compile`, `--compile-mode`, `--use-swiglu`, `--use-lp-layernorm`, `--use-softcap`, `--use-strict-bf16`, `--use-gradient-checkpointing`, `--profile`. Implementation lives in `fast_scgpt/` (e.g. `lp_layernorm.py`, `strict_bf16.py`) and `train()`; see `train_on_modal` / `main` in `modal_train.py`. |
| **`modal_train_distributed.py`** | Per-GPU micro-batch is `batch_size`; effective global batch ≈ `batch_size × gradient_accumulation_steps × num_gpus`. |
| **Local CPU/GPU (no Modal)** | After installing with the `slaf` extra if you need SLAF I/O: `python -m fast_scgpt.train --slaf_path /path/to/dataset.slaf` |

Defaults differ between scripts (for example distributed defaults to a larger `model_size` and `max_genes`); override explicitly for comparable runs.

## Benchmarks

_Coming soon._

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE.md](LICENSE.md).
