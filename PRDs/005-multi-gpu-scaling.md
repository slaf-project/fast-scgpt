# PRD: Multi-GPU Scaling (Phase 3)

**Date:** 2026-02-05
**Status:** Planning
**Prerequisite:** PRD-004 (Architecture Modernization) - Partially Complete

---

## Context

### Current Single-GPU Performance (H100)

| Metric | Value |
|--------|-------|
| GPU | H100-80GB |
| Model | base (100M params) |
| Config | batch=64, max_genes=1024 |
| Step time | 444ms (median) |
| Throughput | 144 cells/sec |
| Memory util | 71% |
| Attention | FA3 native (B,T,H,D) layout |

### Optimization Journey

| Stage | GPU | Step Time | Throughput | Key Change |
|-------|-----|-----------|------------|------------|
| Baseline | A100 | 1360ms | 47 cells/sec | SDPA |
| H100 | H100 | 700ms | 90 cells/sec | Better GPU |
| FA3 native | H100 | 580ms | 110 cells/sec | Zero transpose |
| Vectorized mask | H100 | **444ms** | **144 cells/sec** | Eliminate for-loops |

**3x improvement** on single GPU. Further single-GPU gains are limited.

### What Didn't Work

| Innovation | Result | Reason |
|------------|--------|--------|
| RMSNorm | 33% slower | Doesn't fuse with FA3 kernels |
| ReLU² | 54% slower + 21% more memory | Two-op sequence doesn't fuse |
| QK-Norm | 2x slower | F.normalize breaks kernel fusion |
| Explicit flash_attn on A100 | Slower than SDPA | Transpose overhead |
| Gradient checkpointing | 126 cells/sec at batch=128 | Not worth it for speed |

### Why Multi-GPU?

- Single H100 maxed at 144 cells/sec
- Target: 1000+ cells/sec for production training
- 8x H100 could achieve ~1000 cells/sec (with ~85% scaling efficiency)
- Tahoe100M dataset: 100M cells → would take ~8 days at 144 cells/sec

---

## Goals

1. Scale to 8x H100 with near-linear throughput scaling
2. Achieve 1000+ cells/sec sustained throughput
3. Maintain training stability at large effective batch sizes
4. Enable training on full Tahoe100M dataset in <24 hours

---

## Architecture Options

### Option 1: HuggingFace Accelerate (Recommended)

**Pros:**
- Simple API, minimal code changes
- Handles DDP/FSDP automatically
- Good Modal integration
- Mixed precision built-in

**Cons:**
- Some overhead vs raw PyTorch DDP
- Less control over communication patterns

**Implementation:**
```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

### Option 2: PyTorch DDP (Native)

**Pros:**
- Maximum control
- Lowest overhead
- Well-documented

**Cons:**
- More boilerplate
- Manual gradient sync handling

**Implementation:**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])
```

### Option 3: PyTorch FSDP (For Larger Models)

**Pros:**
- Shards model across GPUs
- Enables much larger models
- Good for 1B+ param models

**Cons:**
- More complex
- Overhead for smaller models
- Not needed for 100M param model

**Recommendation:** Start with Accelerate for simplicity. Switch to DDP if we need more control.

---

## Modal Multi-Node Setup

### Cluster Configuration

```python
import modal

@app.function(
    gpu="H100:8",  # 8x H100 in single node
    # OR for multi-node:
    # gpu="H100",
    # @modal.experimental.clustered(size=8, rdma=True)
)
def train_distributed():
    ...
```

### Reference: Modal nanoGPT Guide

From [modal-labs/multinode-training-guide](https://github.com/modal-labs/multinode-training-guide/tree/main/nanoGPT):

```python
@modal.experimental.clustered(size=N, rdma=True)
def train():
    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank
    world_size = cluster_info.world_size

    # Use torchrun for distributed launch
    os.environ["MASTER_ADDR"] = cluster_info.master_addr
    os.environ["MASTER_PORT"] = str(cluster_info.master_port)
```

---

## Implementation Plan

### Step 1: Single-Node Multi-GPU (8x H100)

**Changes:**
- Add `train_distributed.py` with Accelerate
- Modify `modal_train.py` to support `gpu="H100:8"`
- Test on 8x H100 single node

**Expected:**
- 8x throughput with ~85% efficiency = ~1000 cells/sec
- Same memory per GPU (batch=64 per GPU)

### Step 2: Distributed DataLoader

**Changes:**
- Shard SLAF data across workers
- Each worker loads different cells
- No duplication of data

**Implementation:**
```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, sampler=sampler)
```

### Step 3: Gradient Synchronization

**Automatic with Accelerate/DDP:**
- All-reduce gradients across GPUs
- Synchronized optimizer step
- Effective batch = batch_per_gpu * num_gpus = 64 * 8 = 512

### Step 4: Multi-Node Scaling (If Needed)

**For >8 GPUs:**
- Use Modal clustered decorator
- RDMA for fast inter-node communication
- Up to 64 H100s per cluster

---

## Scaling Projections

| Config | GPUs | Batch/GPU | Effective Batch | Est. Throughput |
|--------|------|-----------|-----------------|-----------------|
| Current | 1 | 64 | 64 | 144 cells/sec |
| Single node | 8 | 64 | 512 | ~1,000 cells/sec |
| Multi-node | 16 | 64 | 1024 | ~1,800 cells/sec |
| Multi-node | 32 | 64 | 2048 | ~3,200 cells/sec |

**Assumptions:** 85% scaling efficiency, no communication bottlenecks

---

## Training Time Estimates

### Tahoe100M Dataset (100M cells)

| Config | Throughput | Time for 1 Epoch |
|--------|------------|------------------|
| 1x H100 | 144 cells/sec | ~8 days |
| 8x H100 | 1,000 cells/sec | ~28 hours |
| 32x H100 | 3,200 cells/sec | ~9 hours |

---

## Cost Estimates

| Config | GPUs | Cost/hr | Time | Total Cost |
|--------|------|---------|------|------------|
| 1x H100 | 1 | $4.32 | 192 hrs | ~$830 |
| 8x H100 | 8 | $34.56 | 28 hrs | ~$970 |
| 32x H100 | 32 | $138.24 | 9 hrs | ~$1,245 |

**Trade-off:** More GPUs = faster but slightly more expensive per epoch.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `fast_scgpt/train_distributed.py` | CREATE | Accelerate-based training |
| `modal_train_distributed.py` | CREATE | Modal multi-GPU wrapper |
| `fast_scgpt/data/distributed.py` | CREATE | Distributed data loading |
| `modal_train.py` | MODIFY | Add single-node multi-GPU option |

---

## Success Criteria

- [ ] 8x H100 achieves 1000+ cells/sec
- [ ] Scaling efficiency >80%
- [ ] Training loss matches single-GPU baseline
- [ ] No OOM on any worker
- [ ] Clean gradient synchronization (no NaN/Inf)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Communication bottleneck | Use NCCL, RDMA on multi-node |
| Data loading bottleneck | Pre-shard data, async loading |
| Large batch instability | Learning rate warmup, gradient clipping |
| Uneven batch sizes | Pad or drop last batch |
| Modal cluster availability | Test on single-node first |

---

## References

- [HuggingFace Accelerate Docs](https://huggingface.co/docs/accelerate)
- [Modal Multi-Node Training Guide](https://github.com/modal-labs/multinode-training-guide)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Nanochat Discussion](https://github.com/karpathy/nanochat/discussions/481)

---

## Appendix: Single-GPU Optimization Summary

### What Worked

1. **H100 over A100**: 2x faster (better memory bandwidth)
2. **FA3 native layout**: 17% faster (zero transpose)
3. **Vectorized mask creation**: 24% faster (eliminate Python loops)
4. **torch.compile**: Marginal gains, helps with kernel fusion
5. **bf16 mixed precision**: Already enabled, essential for speed

### What Didn't Work

1. **RMSNorm/ReLU²/QK-Norm**: All caused regressions due to kernel fusion issues
2. **Explicit flash_attn on A100**: Transpose overhead > SDPA auto-dispatch
3. **Gradient checkpointing for speed**: Only useful for fitting larger batches

### Lessons Learned

- PyTorch's SDPA is highly optimized; explicit flash_attn often slower
- Nanochat innovations assume torch.compile fuses everything; doesn't always work
- Profile before optimizing - mask creation was 26% of step time
- Native tensor layouts matter more than algorithm choice
