# Fast-scGPT Implementation Plan

## Overview

Reimplement scGPT from scratch incorporating modded-nanogpt innovations, using HuggingFace Accelerate for distributed training. Following "make it work, make it right, make it fast" philosophy.

**First Milestone**: Train a small transformer on MPS backend with SLAF data, observe decreasing loss.

---

## Phase 1: "Make it Work" — Minimal Viable Training Loop

**Goal**: Get a transformer training on MPS with decreasing loss using SLAF infrastructure.

### 1.1 Project Setup

Create project structure:
```
fast-scgpt/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── pytest.ini
├── .coveragerc
├── PRDs/
│   └── implementation-plan.md
├── fast_scgpt/
│   ├── __init__.py
│   ├── model.py          # Minimal transformer
│   ├── config.py         # Model configuration
│   ├── train.py          # Training loop
│   └── device.py         # MPS compatibility
└── tests/
    ├── __init__.py
    ├── test_model.py
    └── test_training.py
```

Dependencies: `torch>=2.0`, `einops`, `numpy`, `slafdb` (from existing SLAF project)

### 1.2 SLAF Data Integration

Use existing infrastructure at `/Users/pavan/slaf-project/slaf/`:

```python
from slaf import SLAFArray
from slaf.ml import SLAFDataLoader

slaf_array = SLAFArray("../slaf-datasets/plate1_Tahoe100M_v21.slaf")
dataloader = SLAFDataLoader(
    slaf_array=slaf_array,
    tokenizer_type="scgpt",
    batch_size=32,
    max_genes=512,              # Start small
    n_expression_bins=10,       # SLAF default (scGPT paper uses 51)
    use_mixture_of_scanners=True,
)
```

**Output format** from SLAF:
- `input_ids`: `[CLS] gene1 expr1 gene2 expr2 ... [SEP] [PAD]` (shape: batch × 2050)
- `attention_mask`: 1 for real tokens, 0 for padding
- `cell_ids`: integer cell IDs

**Special tokens**: PAD=0, CLS=1, SEP=2, MASK=3, genes start at 4

### 1.3 Minimal Model (4 layers, 4 heads, 256 dim)

- **TokenEmbedding**: gene tokens → embeddings (vocab ~50k genes + special tokens)
- **ExpressionEmbedding**: binned expression (10 bins) → embeddings
- **TransformerBlock**: standard attention + FFN
- **Output heads**: predict masked gene tokens and expression bins

Simplifications for Phase 1:
- Standard LayerNorm (not RMSNorm yet)
- GELU activation (not ReLU² yet)
- AdamW optimizer (not Muon yet)

### 1.4 Training Loop

- Load batches from SLAFDataLoader
- Parse interleaved format: odd positions = genes, even positions = expressions
- Forward pass: sum gene + expression embeddings at each position
- Random masking (15-30% of genes)
- Loss: cross-entropy on masked positions
- Log loss every N steps

### 1.5 MPS Compatibility

- Device detection (MPS > CUDA > CPU)
- Tensors from SLAF are CPU; move to device in training loop
- No Flash Attention (not available on MPS)

### Success Criteria
- [x] Model forward pass works on MPS
- [x] 50 training steps complete without error on SLAF data (Tahoe100M)
- [x] Loss decreases over training (12.20 → 10.76)
- [x] `pytest tests/` passes (20/20 tests)

---

## Phase 2: "Make it Right" — Full Architecture + Innovations

**Goal**: Implement full scGPT architecture with modded-nanogpt innovations.

### 2.1 Full Transformer Architecture

Scale to scGPT specs: 12 layers, 8 heads, 512 dim

Add modded-nanogpt innovations:
| Innovation | Description |
|------------|-------------|
| **QK-Norm** | Normalize Q and K before attention scores |
| **ReLU²** | Replace GELU with ReLU squared in FFN |
| **RMSNorm** | Replace LayerNorm, no learnable params |

**Note on RoPE**: scGPT genes have no inherent sequential order. Skip RoPE initially; may test "partial RoPE" on interleaved positions later.

### 2.2 Custom Attention Masking

scGPT uses specialized bidirectional masking:
- Genes attend to all other genes in same cell (not causal)
- Expression tokens attend to their gene token
- Support for masked gene prediction

### 2.3 Expression Bins Configuration

SLAF default is 10 bins; scGPT paper uses 51 bins. Two options:
1. **Match SLAF default** (10 bins) - simpler integration
2. **Configure SLAF** for 51 bins - matches paper exactly

```python
dataloader = SLAFDataLoader(
    ...,
    n_expression_bins=51,  # Match scGPT paper
)
```

### 2.4 Masked Gene Expression Prediction Loss

Training objective matching scGPT paper:
- Randomly mask 15-30% of genes
- Predict both gene identity and expression bin
- Multi-task loss combining both objectives

### Success Criteria
- [ ] Model parameter count ~53M (matching scGPT)
- [ ] Training on SLAF data shows decreasing loss
- [ ] Attention masking visualizations look correct
- [ ] QK-norm and ReLU² active and numerically stable

---

## Phase 3: "Make it Fast" — CUDA, Flash Attention, Distributed

**Goal**: Optimize for CUDA GPUs, integrate Flash Attention, enable multi-node training.

### 3.1 CUDA Backend + Flash Attention

**Design for CUDA from the start** - MPS is for local dev/debug only.

**Flash Attention integration**:
```python
# Flash Attention 2 (broader support: A100, RTX 3090/4090, etc.)
from flash_attn import flash_attn_func

# Flash Attention 3 (H100/H800 only, CUDA 12.3+, beta)
# Use when training on Modal H100 clusters
```

**Attention module with fallback**:
```python
def attention(q, k, v, mask=None):
    if FLASH_ATTN_AVAILABLE and q.device.type == "cuda":
        return flash_attn_func(q, k, v, causal=False)  # bidirectional
    else:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

Requirements: `flash-attn` (pip install), CUDA toolkit, `ninja` build tool

### 3.2 Muon Optimizer

Implement Muon (Momentum Orthogonalized by Newton-Schulz):
- Separate handling for embeddings (AdamW) vs weights (Muon)
- Cautious weight decay (only decay where grad*param >= 0)
- Polar Express variant for faster convergence

### 3.3 Memory Optimizations

- Gradient checkpointing for large models
- Mixed precision (bf16 on CUDA, fp32 fallback on MPS)
- `torch.compile` for fused kernels (CUDA)

### 3.4 HuggingFace Accelerate Integration

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

Support: single GPU → multi-GPU DDP → FSDP

### 3.5 Modal Multi-Node Training

Reference: https://github.com/modal-labs/multinode-training-guide/tree/main/nanoGPT

**Modal cluster setup**:
```python
import modal

@modal.experimental.clustered(size=4, rdma=True)
def train_cluster():
    cluster_info = modal.experimental.get_cluster_info()
    # torchrun integration with cluster_info.rank, cluster_info.world_size
```

**Key patterns**:
- Use `torchrun` for distributed launch
- RDMA enabled for fast inter-node communication
- Up to 64 H100 SXM devices per cluster
- Flash Attention 3 on H100s

### Success Criteria
- [ ] Flash Attention 2 works on CUDA (A100/RTX series)
- [ ] Flash Attention 3 works on H100 (Modal)
- [ ] Muon matches or beats AdamW convergence
- [ ] DDP works on multi-GPU (local) and multi-node (Modal)
- [ ] Training throughput ≥10,000 cells/sec on H100

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Gene embeddings | Sum gene + expression | Matches scGPT paper |
| Attention | Bidirectional (not causal) | Genes are unordered sets |
| RoPE | Skip initially | No sequential order for genes |
| Optimizer progression | AdamW → Muon | Start stable, optimize later |
| Expression bins | 51 bins | Matches scGPT default |
| Attention impl | Flash Attn → SDPA fallback | Design for CUDA, fallback for MPS |
| Compute strategy | MPS dev → CUDA prod → Modal scale | Local debug, GPU train, cluster scale |

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| MPS op not supported | Test each component early; fallback to CPU |
| Loss not decreasing | Start with overfitting on tiny data; verify gradients |
| Attention masking incorrect | Unit tests + visualizations; compare to reference |
| ReLU² instability | Monitor activation stats; gradient clipping |
| Muon diverges | Start with AdamW baseline; careful tuning |
| Flash Attention build fails | Use prebuilt wheels; fallback to SDPA |
| Modal cluster access | Contact Modal via Slack; test on single GPU first |

---

## Verification Plan

1. **Phase 1**: Run `python -m fast_scgpt.train` on MPS, observe loss going down
2. **Phase 2**: Train on pbmc3k SLAF dataset, verify architecture innovations work
3. **Phase 3a**: Single CUDA GPU with Flash Attention 2 → benchmark throughput
4. **Phase 3b**: Modal multi-node with Flash Attention 3 → scale to full dataset

---

## Progress Checklist

### Phase 1: Make it Work

- [x] 1.1 Project scaffolding (pyproject.toml, .gitignore, etc.)
- [x] 1.2 Model configuration dataclass
- [x] 1.3 Device detection module
- [x] 1.4 Minimal transformer model
- [x] 1.5 Training loop with SLAF integration
- [x] 1.6 Model unit tests
- [ ] 1.7 Verify loss decreases on MPS

### Phase 2: Make it Right

- [ ] 2.1 Scale to full scGPT architecture (12 layers, 512 dim)
- [ ] 2.2 Add QK-Norm to attention
- [ ] 2.3 Replace GELU with ReLU²
- [ ] 2.4 Replace LayerNorm with RMSNorm
- [ ] 2.5 Implement custom attention masking
- [ ] 2.6 Configure 51 expression bins
- [ ] 2.7 Verify ~53M parameters

### Phase 3: Make it Fast

- [ ] 3.1 Flash Attention integration with SDPA fallback
- [ ] 3.2 Muon optimizer implementation
- [ ] 3.3 Accelerate DDP integration
- [ ] 3.4 Modal multi-node training script
- [ ] 3.5 Benchmark throughput on H100

---

## Atomic Commits Plan

### Phase 1 Commits

```bash
/atomic-commit chore "initialize project with uv and pre-commit"
/atomic-commit feature "add model configuration dataclass"
/atomic-commit feature "add MPS/CUDA/CPU device detection"
/atomic-commit feature "implement minimal transformer model"
/atomic-commit feature "add training loop with SLAF loader"
/atomic-commit tests "add model forward pass tests"
/atomic-commit docs "add implementation plan to PRDs"
```

### Phase 2 Commits

```bash
/atomic-commit feature "scale model to 12 layers 512 dim"
/atomic-commit feature "add QK-Norm to attention module"
/atomic-commit feature "replace GELU with ReLU squared"
/atomic-commit feature "replace LayerNorm with RMSNorm"
/atomic-commit feature "implement bidirectional attention mask"
/atomic-commit tests "add architecture validation tests"
```

### Phase 3 Commits

```bash
/atomic-commit feature "add Flash Attention with SDPA fallback"
/atomic-commit feature "implement Muon optimizer"
/atomic-commit feature "add Accelerate DDP training script"
/atomic-commit feature "add Modal multi-node training"
/atomic-commit perf "enable torch.compile and bf16"
/atomic-commit tests "add distributed training tests"
```

---

## Files to Create

### Phase 1

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project metadata |
| `uv.lock` | Locked dependencies (generated by uv) |
| `.gitignore` | Git ignore patterns |
| `pytest.ini` | Pytest configuration |
| `.coveragerc` | Coverage configuration |
| `PRDs/implementation-plan.md` | This plan document |
| `fast_scgpt/__init__.py` | Package init |
| `fast_scgpt/config.py` | Model configuration dataclass |
| `fast_scgpt/model.py` | Core transformer implementation |
| `fast_scgpt/train.py` | Training loop entry point |
| `fast_scgpt/device.py` | MPS/CUDA/CPU device handling |
| `tests/__init__.py` | Tests package init |
| `tests/test_model.py` | Model smoke tests |

### Phase 3

| File | Purpose |
|------|---------|
| `fast_scgpt/attention.py` | Flash Attention integration with SDPA fallback |
| `fast_scgpt/optim/muon.py` | Muon optimizer implementation |
| `fast_scgpt/train_distributed.py` | Accelerate-based distributed training |
| `modal_train.py` | Modal multi-node training script |

## SLAF Integration Reference

Key files in `/Users/pavan/slaf-project/slaf/slaf/ml/`:

| File | Key Classes | Purpose |
|------|-------------|---------|
| `dataloaders.py` | `SLAFDataLoader` | Streaming batches with scGPT tokenization |
| `tokenizers.py` | `SLAFTokenizer` | Gene → token ID mapping |
| `aggregators.py` | `ScGPTWindow` | Expression binning (log1p → bins) |
| `datasets.py` | `SLAFIterableDataset` | PyTorch IterableDataset wrapper |
| `samplers.py` | `RandomShuffle` | Cell shuffling strategies |
