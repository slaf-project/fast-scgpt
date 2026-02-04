# PRD: A100 + Flash Attention Upgrade

**Date:** 2026-02-04
**Status:** Planning
**Prerequisite:** PRD-002 (L4 memory optimization) ✅ Complete

---

## Context

### L4 Baseline (from PRD-002)
| Metric | Value |
|--------|-------|
| GPU | L4 (24GB, $0.99/hr) |
| Best config | batch=16, max_genes=256 |
| Memory util | 64% |
| Step time | ~400ms (fwd=143ms, bwd=217ms) |
| Throughput | ~85 cells/sec |
| Model | small (35M params) |

### Why A100?
1. L4 maxes out at batch=16, max_genes=256 (OOM at higher)
2. Base model (53M params) doesn't fit on L4 at reasonable batch size
3. Flash Attention requires sm80+ (A100=sm80, L4=sm89 but slower)
4. A100 has 2x memory bandwidth (1.5 TB/s vs 300 GB/s)

---

## Goals

1. Run base model (53M params) with batch=32+, max_genes=512
2. Integrate Flash Attention 2 for memory-efficient attention
3. Benchmark and compare: L4 baseline vs A100 vs A100+FlashAttn
4. Establish throughput targets for production training

---

## Implementation Plan

### Step 1: Switch to A100 (no code changes)

**Changes:**
- `modal_train.py`: Change `gpu="L4"` to `gpu="A100"`

**Test:**
```bash
# Same config as L4 baseline - expect 2-3x speedup
modal run modal_train.py --batch-size 16 --max-genes 256 --n-steps 20 --profile

# Push batch size (A100 has 40GB vs L4's 24GB)
modal run modal_train.py --batch-size 32 --max-genes 256 --n-steps 20 --profile

# Target config
modal run modal_train.py --batch-size 32 --max-genes 512 --n-steps 20 --profile
```

**Success criteria:**
- [ ] batch=32, max_genes=512 fits without OOM
- [ ] Step time < 300ms (vs L4's 400ms)
- [ ] Memory util < 80%

### Step 2: Add Flash Attention 2

**Changes:**
1. Add `flash-attn` to Modal image dependencies
2. Create `fast_scgpt/attention.py` with FA2 + SDPA fallback
3. Update `MultiHeadAttention` to use new attention module

**attention.py design:**
```python
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

def attention(q, k, v, mask=None, causal=False):
    """Memory-efficient attention with Flash Attention fallback."""
    if FLASH_ATTN_AVAILABLE and q.is_cuda:
        # Flash Attention 2: (batch, seqlen, nheads, headdim)
        return flash_attn_func(q, k, v, causal=causal)
    else:
        # PyTorch SDPA fallback
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

**Test:**
```bash
# Compare with/without Flash Attention
modal run modal_train.py --batch-size 32 --max-genes 512 --n-steps 20 --profile
# Check: fwd time should decrease, memory should decrease
```

**Success criteria:**
- [ ] Flash Attention loads without error on A100
- [ ] Forward pass 20-30% faster
- [ ] Memory usage 20-30% lower (can increase batch size)

### Step 3: Test Base Model (53M params)

**Changes:**
- Use `--model-size base` flag

**Test:**
```bash
# Base model on A100 with Flash Attention
modal run modal_train.py --model-size base --batch-size 32 --max-genes 512 --n-steps 20 --profile
```

**Success criteria:**
- [ ] Base model fits with batch=32, max_genes=512
- [ ] Loss decreases (sanity check)
- [ ] Throughput > 100 cells/sec

### Step 4: Optimize and Benchmark

**Experiments:**
1. Find max batch size at 80% memory
2. Compare torch.compile effect on A100
3. Test gradient checkpointing (if needed for larger configs)

**Target metrics:**
| Config | Target |
|--------|--------|
| batch_size | 64+ |
| max_genes | 512 |
| seq_len | 1026 |
| Throughput | 200+ cells/sec |
| Memory util | 70-80% |

---

## Cost Estimate

| GPU | Cost/hr | Est. benchmark time | Est. cost |
|-----|---------|---------------------|-----------|
| A100-40GB | $3.00 | 30 min | ~$1.50 |
| A100-80GB | $4.00 | 30 min | ~$2.00 |

Start with A100-40GB, upgrade to 80GB only if needed.

---

## Files to Modify

| File | Change |
|------|--------|
| `modal_train.py` | Change gpu="L4" to gpu="A100" |
| `pyproject.toml` | Add flash-attn to cuda extras |
| `fast_scgpt/attention.py` | NEW: Flash Attention wrapper |
| `fast_scgpt/model.py` | Use new attention module |

---

## Rollback Plan

If Flash Attention causes issues:
1. SDPA fallback is built-in (no code change needed)
2. Can revert to L4 by changing gpu= parameter

---

## References

- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [flash-attn PyPI](https://pypi.org/project/flash-attn/)
- [Modal GPU Pricing](https://modal.com/pricing)
- [A100 Specs](https://www.nvidia.com/en-us/data-center/a100/)
