# Fast-scGPT

Fast reimplementation of scGPT incorporating modded-nanogpt innovations.

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Training

```bash
python -m fast_scgpt.train --slaf_path ../slaf-datasets/plate1_Tahoe100M_v21.slaf
```
