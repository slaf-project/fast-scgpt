# PRD: Architecture Modernization (Phase 2)

**Date:** 2026-02-04
**Status:** Planning
**Prerequisite:** PRD-003 (A100 + Flash Attention 2) ✅ Complete

---

## Context

### Current Baseline (from PRD-003)

| Metric | Value |
|--------|-------|
| GPU | A100-80GB |
| Model | base (100M params) |
| Config | batch=64, max_genes=1024 |
| Step time | 1.36s |
| Throughput | 47 cells/sec |
| Memory util | 71% |
| Attention | Flash Attention 2 |

### Why Modernize?

The [nanochat speedrun](https://github.com/karpathy/nanochat/discussions/481) achieved **600x cost reduction** vs original GPT-2 training through architectural innovations. Many of these apply directly to scGPT:

- **RMSNorm**: Faster than LayerNorm, no learnable parameters
- **ReLU²**: Sparser activations, computationally cheaper than GELU
- **QK-Norm**: Stabilizes attention without softcapping
- **Muon optimizer**: Faster convergence than AdamW for weight matrices

---

## Goals

1. Implement proven nanochat innovations applicable to scGPT
2. Measure impact on training speed and convergence
3. Prepare infrastructure for H100 scaling (FA3, FP8)
4. Establish new throughput baseline for Phase 3 distributed training

---

## Innovation Analysis

### Tier 1: High Priority (Proven, Easy to Implement)

| Innovation | Description | Expected Benefit | Complexity |
|------------|-------------|------------------|------------|
| **RMSNorm** | Root mean square norm, no learnable γ/β | 10-15% faster norm ops | Low |
| **ReLU²** | `F.relu(x).square()` instead of GELU | Sparse activations, faster | Low |
| **QK-Norm** | Normalize Q, K after projection | Stable attention, no softcap needed | Low |

### Tier 2: Medium Priority (Optimizer Changes)

| Innovation | Description | Expected Benefit | Complexity |
|------------|-------------|------------------|------------|
| **Muon** | Momentum orthogonalized by Newton-Schulz | Faster convergence | Medium |
| **Split optimizer** | AdamW for embeddings, Muon for weights | Better per-param optimization | Medium |
| **Untied embeddings** | Separate input/output embeddings | Better learning dynamics | Low |

### Tier 3: Lower Priority (Experimental)

| Innovation | Description | Expected Benefit | Complexity |
|------------|-------------|------------------|------------|
| **Logit softcapping** | `15 * tanh(logits / 15)` | Prevents extreme outputs | Low |
| **Value embeddings** | Gated V additions at alternating layers | +expressivity, minimal overhead | Medium |
| **Per-layer residual scalars** | Learnable `λ` for residual weighting | Flexible depth utilization | Low |

### Tier 4: H100-Specific Optimizations

| Innovation | Description | Expected Benefit | Complexity |
|------------|-------------|------------------|------------|
| **Flash Attention 3** | H100-native, tensor layout optimization | 1.5-2x faster attention | Medium |
| **FP8 training** | 8-bit floating point | 2x memory, faster matmuls | High |

**Note on FP8:** The nanochat discussion lists FP8 as an "unsuccessful approach" for their use case. However, this may be task-dependent. We should benchmark but not prioritize.

### Not Applicable to scGPT

| Innovation | Reason to Skip |
|------------|----------------|
| **RoPE** | Genes have no sequential order (unordered sets) |
| **Sliding window attention** | Need full bidirectional attention for gene-gene interactions |
| **Causal masking** | scGPT uses bidirectional (masked LM style) |

---

## Implementation Plan

### Step 1: RMSNorm (Replace LayerNorm)

**Changes:**
- Create `fast_scgpt/norms.py` with RMSNorm implementation
- Replace all `nn.LayerNorm` in `model.py`

**Implementation:**
```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable parameters)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # No learnable gamma/beta - just normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms
```

**Test:**
```bash
modal run modal_train.py --model-size base --batch-size 64 --max-genes 1024 --n-steps 50 --profile
# Compare step time vs LayerNorm baseline
```

**Success criteria:**
- [ ] Step time decreases (expect 5-10%)
- [ ] Loss still decreases normally
- [ ] No numerical instability

---

### Step 2: ReLU² (Replace GELU)

**Changes:**
- Update `FeedForward` in `model.py`

**Implementation:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x)
    x = F.relu(x).square()  # ReLU² instead of GELU
    x = self.dropout(x)
    x = self.fc2(x)
    return x
```

**Test:**
```bash
modal run modal_train.py --model-size base --batch-size 64 --max-genes 1024 --n-steps 100 --profile
# Monitor activation sparsity and loss curve
```

**Success criteria:**
- [ ] Comparable or better convergence than GELU
- [ ] Faster forward pass (fewer ops than GELU approximation)
- [ ] Sparse activations (many zeros after ReLU)

---

### Step 3: QK-Norm (Stabilize Attention)

**Changes:**
- Add normalization after Q, K projections in `MultiHeadAttention`

**Implementation:**
```python
def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # Reshape to (batch, n_heads, seq_len, d_head)
    q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

    # QK-Norm: normalize Q and K per head
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    # ... rest of attention
```

**Success criteria:**
- [ ] More stable attention scores (no NaN/Inf)
- [ ] Can remove any existing softcapping/clipping
- [ ] Loss converges smoothly

---

### Step 4: Muon Optimizer

**Changes:**
- Create `fast_scgpt/optim/muon.py`
- Update `train.py` to use split optimizer strategy

**Implementation (simplified):**
```python
class Muon(torch.optim.Optimizer):
    """Momentum Orthogonalized by Newton-Schulz."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)

                # Orthogonalize via Newton-Schulz (for 2D weight matrices)
                if g.dim() == 2:
                    g = self._newton_schulz(buf if group['nesterov'] else g)
                else:
                    g = buf if group['nesterov'] else g

                p.add_(g, alpha=-group['lr'])

    def _newton_schulz(self, G, steps=5):
        """Orthogonalize gradient via Newton-Schulz iteration."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.float() / (G.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * A @ X + c * A @ A @ X
        return X.to(G.dtype)
```

**Split optimizer strategy:**
```python
# Separate parameter groups
embed_params = [p for n, p in model.named_parameters() if 'embedding' in n]
weight_params = [p for n, p in model.named_parameters() if 'embedding' not in n and p.dim() == 2]
other_params = [p for n, p in model.named_parameters() if 'embedding' not in n and p.dim() != 2]

optimizer = SplitOptimizer([
    {'params': embed_params, 'optimizer': 'adamw', 'lr': 1e-4},
    {'params': weight_params, 'optimizer': 'muon', 'lr': 0.02},
    {'params': other_params, 'optimizer': 'adamw', 'lr': 1e-4},
])
```

**Success criteria:**
- [ ] Faster convergence than pure AdamW (fewer steps to same loss)
- [ ] No divergence or instability
- [ ] Works with gradient accumulation

---

### Step 5: Flash Attention 3 (H100 Only)

**Changes:**
- Update `attention.py` to detect H100 and use FA3
- Update Modal image for H100 + FA3

**Implementation:**
```python
# Check for FA3 availability (H100 only)
FLASH_ATTN_3_AVAILABLE = False
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    FLASH_ATTN_3_AVAILABLE = True
except ImportError:
    flash_attn_3_func = None

def attention_with_reshape(q, k, v, ...):
    if FLASH_ATTN_3_AVAILABLE and q.is_cuda:
        # FA3 native tensor layout
        return flash_attn_3_func(q, k, v, causal=False)
    elif FLASH_ATTN_AVAILABLE and q.is_cuda:
        # FA2 fallback
        ...
```

**Modal H100 config:**
```python
@app.function(
    gpu="H100",  # or "H100:8" for multi-GPU
    ...
)
```

**Success criteria:**
- [ ] FA3 loads on H100 without error
- [ ] 1.5-2x faster attention vs FA2
- [ ] Same numerical results as FA2

---

### Step 6: FP8 Training (Experimental)

**Note:** Nanochat found FP8 unsuccessful. We'll benchmark but not prioritize.

**Changes:**
- Add FP8 option to training config
- Use `torch.float8_e4m3fn` for forward, `torch.float8_e5m2` for backward

**Test:**
```bash
modal run modal_train.py --model-size base --batch-size 128 --precision fp8 --n-steps 50
```

**Success criteria:**
- [ ] 2x memory reduction (can double batch size)
- [ ] Convergence matches bf16 baseline
- [ ] No significant accuracy loss

---

## Rollout Strategy

### Phase 2a: Architecture (Steps 1-3)
```bash
# Incremental testing
# 1. RMSNorm only
modal run modal_train.py --n-steps 100 --profile

# 2. RMSNorm + ReLU²
modal run modal_train.py --n-steps 100 --profile

# 3. RMSNorm + ReLU² + QK-Norm
modal run modal_train.py --n-steps 100 --profile
```

### Phase 2b: Optimizer (Step 4)
```bash
# Compare optimizers
modal run modal_train.py --optimizer adamw --n-steps 500
modal run modal_train.py --optimizer muon --n-steps 500
# Plot loss curves, compare convergence
```

### Phase 2c: H100 Scaling (Steps 5-6)
```bash
# Upgrade to H100
modal run modal_train.py --gpu h100 --n-steps 100 --profile
# If stable, test FA3
modal run modal_train.py --gpu h100 --flash-attn-version 3 --n-steps 100
```

---

## Target Metrics

| Metric | Current (A100+FA2) | Target (A100+Modernized) | Target (H100+FA3) |
|--------|-------------------|--------------------------|-------------------|
| Step time | 1.36s | <1.0s | <0.5s |
| Throughput | 47 cells/sec | >60 cells/sec | >150 cells/sec |
| Memory util | 71% | 70-80% | 70-80% |
| Convergence | baseline | 20% fewer steps | 20% fewer steps |

---

## Cost Estimate

| Phase | GPU | Est. Time | Est. Cost |
|-------|-----|-----------|-----------|
| 2a (Architecture) | A100-80GB | 1 hour | ~$4 |
| 2b (Optimizer) | A100-80GB | 2 hours | ~$8 |
| 2c (H100 scaling) | H100 | 1 hour | ~$4.50 |
| **Total** | | ~4 hours | ~$16.50 |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `fast_scgpt/norms.py` | CREATE | RMSNorm implementation |
| `fast_scgpt/model.py` | MODIFY | Use RMSNorm, ReLU², QK-Norm |
| `fast_scgpt/optim/__init__.py` | CREATE | Optimizer package |
| `fast_scgpt/optim/muon.py` | CREATE | Muon optimizer |
| `fast_scgpt/attention.py` | MODIFY | Add FA3 support |
| `fast_scgpt/train.py` | MODIFY | Split optimizer, new flags |
| `modal_train.py` | MODIFY | Add --optimizer, --precision flags |
| `tests/test_norms.py` | CREATE | RMSNorm tests |
| `tests/test_muon.py` | CREATE | Muon optimizer tests |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| ReLU² causes dead neurons | Monitor activation stats; fallback to GELU |
| Muon diverges | Start with AdamW warmup; tune learning rate |
| QK-Norm changes attention dynamics | Compare attention patterns; tune scale |
| FA3 not available on Modal | Fallback to FA2 (already working) |
| FP8 causes accuracy loss | Keep as experimental; don't ship if broken |

---

## References

- [Nanochat Speedrun Discussion](https://github.com/karpathy/nanochat/discussions/481)
- [Flash Attention 3 Paper](https://arxiv.org/abs/2407.08608)
- [Muon Optimizer (modded-nanogpt)](https://github.com/KellerJordan/modded-nanogpt)
- [RMSNorm Paper](https://arxiv.org/abs/1910.07467)
- [QK-Norm (Gemma 2)](https://arxiv.org/abs/2408.00118)

---

## Success Criteria Summary

- [ ] RMSNorm replaces LayerNorm, training stable
- [ ] ReLU² replaces GELU, convergence maintained
- [ ] QK-Norm added, attention more stable
- [ ] Muon optimizer shows faster convergence than AdamW
- [ ] Step time reduced by 25%+ on A100
- [ ] H100 + FA3 achieves 3x throughput vs A100 + FA2
- [ ] All changes maintain or improve final loss
