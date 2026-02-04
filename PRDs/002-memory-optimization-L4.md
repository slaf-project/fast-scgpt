# PRD: Memory Optimization for L4 GPU Training

**Date:** 2026-02-04
**Status:** Implementation Complete - Ready for Testing
**Goal:** Maximize batch size and sequence length on L4 (24GB) before upgrading to A100

---

## Current State Summary

### What's Working
- ✅ GPU metrics infrastructure (`GPUMetrics` class)
- ✅ Mixed precision training (bf16 on L4)
- ✅ Modal deployment with `--detach` support
- ✅ Dynamic vocab_size from SLAF metadata
- ✅ Training runs, loss decreases (15.1 → 12.6 in 6 steps)
- ✅ HuggingFace streaming for data loading

### Current Baseline (L4 GPU)
| Parameter | Value |
|-----------|-------|
| GPU | L4 (24GB VRAM) |
| Model params | ~35M (small config) |
| batch_size | 8 |
| max_genes | 64 |
| seq_len | 130 (2 + 64*2) |
| Peak memory | 7.7 GB (33% of 24GB) |
| Step time | ~400ms |
| Throughput | ~20 cells/sec |

### Problem
Memory usage is **much higher than expected**:
- Original estimate: batch_size=32, max_genes=512 → ~2.5GB
- Reality: batch_size=8, max_genes=64 → ~7.7GB

Scaling to production configs causes OOM:
- batch_size=32, max_genes=64 → OOM
- batch_size=8, max_genes=256 → OOM

---

## Memory Analysis

### Where is memory going?

| Component | Estimated Size | Notes |
|-----------|---------------|-------|
| Model params (bf16) | ~70 MB | 35M × 2 bytes |
| Optimizer states | ~280 MB | AdamW: 2× params in fp32 |
| Gradients (bf16) | ~70 MB | Same as params |
| **Activations** | **~7+ GB** | The culprit |

### Activation Memory Breakdown
For batch=8, seq_len=130, d_model=256, n_layers=4:

| Layer | Shape | Size per layer |
|-------|-------|----------------|
| Attention QKV | (8, 130, 768) | 0.8 MB |
| Attention scores | (8, 4, 130, 130) | 2.2 MB |
| Attention output | (8, 130, 256) | 0.3 MB |
| FFN intermediate | (8, 130, 1024) | 1.1 MB |
| **Total per layer** | | ~4.4 MB |
| **4 layers** | | ~18 MB |

This doesn't add up to 7GB. Suspect issues:
1. **PyTorch allocator overhead** - fragmentation
2. **Autocast caching** - mixed precision buffers
3. **Backward pass** - stores activations for gradients
4. **Data loading** - tensors not released

---

## Optimization Plan

### Phase 1: Quick Wins (No Code Changes)

#### 1.1 Memory Allocator Tuning
```python
# Add to modal_train.py before training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

#### 1.2 Gradient Accumulation
Instead of batch_size=32, use batch_size=8 with 4 accumulation steps:
```python
# Effective batch = 32, but memory = batch 8
for i, batch in enumerate(dataloader):
    loss = train_step(...) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Phase 2: Code Optimizations

#### 2.1 Gradient Checkpointing
Trade compute for memory by not storing intermediate activations:
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward, x, mask, use_reentrant=False)
        return self._forward(x, mask)
```
Expected: ~50% memory reduction for activations

#### 2.2 torch.compile
Fused kernels reduce memory fragmentation:
```python
model = torch.compile(model, mode="reduce-overhead")
```
Expected: 10-30% memory reduction, 20-50% speedup

#### 2.3 Attention Optimization
Use memory-efficient attention:
```python
# Option A: PyTorch 2.0 SDPA (already default)
F.scaled_dot_product_attention(q, k, v, is_causal=False)

# Option B: Flash Attention (requires sm80+, L4 is sm89)
from flash_attn import flash_attn_func
```

### Phase 3: Architecture Changes

#### 3.1 Reduce d_model
Current: d_model=256 → Try d_model=128 for memory testing

#### 3.2 Reduce n_layers
Current: n_layers=4 → Try n_layers=2 for memory testing

---

## Implementation Order

| Priority | Optimization | Expected Gain | Effort | Status |
|----------|--------------|---------------|--------|--------|
| 1 | Gradient accumulation | 4x effective batch | Low | ✅ Done |
| 2 | PYTORCH_CUDA_ALLOC_CONF | 10-20% | None | ✅ Done |
| 3 | torch.compile | 10-30% memory, 20-50% speed | Low | ✅ Done |
| 4 | Gradient checkpointing | 50% activation memory | Medium | ✅ Done |
| 5 | Flash Attention | Better for long seqs | Medium | Deferred |

### Implementation Notes

**Gradient Accumulation (2026-02-04):**
```bash
# Use micro-batch of 8, accumulate 4x for effective batch of 32
modal run modal_train.py --batch-size 8 --gradient-accumulation-steps 4
```

**torch.compile (2026-02-04):**
```bash
# Enable fused kernels for potential speedup
modal run modal_train.py --use-compile
```

**Gradient Checkpointing (2026-02-04):**
```bash
# Trade compute for ~50% activation memory savings
modal run modal_train.py --use-gradient-checkpointing
```

**Combined for Maximum Memory Efficiency:**
```bash
modal run modal_train.py \
  --batch-size 8 \
  --max-genes 256 \
  --gradient-accumulation-steps 4 \
  --use-gradient-checkpointing \
  --use-compile
```

---

## Success Criteria

### Target Configuration on L4 (24GB)
| Parameter | Current | Target |
|-----------|---------|--------|
| batch_size | 8 | 32+ |
| max_genes | 64 | 512 |
| seq_len | 130 | 1026 |
| Memory util | 33% | 80-90% |
| Throughput | 20 cells/sec | 100+ cells/sec |

### When to Upgrade to A100
- If L4 can't hit batch_size=32, max_genes=256 after optimizations
- If throughput is bottlenecked by memory bandwidth (not compute)
- For base model (53M params) training

---

## Testing Plan

### Test 1: Baseline (no optimizations)
```bash
modal run modal_train.py --batch-size 8 --max-genes 64 --n-steps 20
# Expected: ~33% memory, ~400ms/step
```

### Test 2: Gradient Accumulation Only
```bash
modal run modal_train.py --batch-size 8 --max-genes 64 --n-steps 20 \
  --gradient-accumulation-steps 4
# Expected: ~33% memory, effective batch=32
```

### Test 3: Gradient Checkpointing Only
```bash
modal run modal_train.py --batch-size 8 --max-genes 128 --n-steps 20 \
  --use-gradient-checkpointing
# Expected: ~20% memory (vs ~40% without), can increase max_genes
```

### Test 4: torch.compile Only
```bash
modal run modal_train.py --batch-size 8 --max-genes 64 --n-steps 20 \
  --use-compile
# Expected: Faster step time, slightly lower memory
```

### Test 5: All Optimizations Combined
```bash
modal run modal_train.py --batch-size 8 --max-genes 256 --n-steps 50 \
  --gradient-accumulation-steps 4 \
  --use-gradient-checkpointing \
  --use-compile
# Target: batch_size=8×4=32, max_genes=256, <80% memory
```

### Test 6: Push to Limits
```bash
modal run modal_train.py --batch-size 16 --max-genes 512 --n-steps 50 \
  --gradient-accumulation-steps 2 \
  --use-gradient-checkpointing \
  --use-compile
# Target: batch_size=16×2=32, max_genes=512, find memory limit
```

---

## Related Files

- `fast_scgpt/train.py` - Training loop, metrics
- `fast_scgpt/model.py` - Model architecture
- `modal_train.py` - Modal deployment
- `PRDs/BUG-slaf-tokenizer-expression-clipping.md` - Tokenizer bug (separate issue)

---

## References

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
