# BUG: SLAF Tokenizer Missing Expression Bin Clipping

**Date:** 2026-02-04
**Severity:** High
**Component:** slaf/slaf/ml/tokenizers.py

## Summary

The SLAF tokenizer assumes integer expression values are pre-binned and adds `expr_bin_start` without clipping, causing token IDs to exceed `total_vocab_size` when raw integer expression counts are passed.

## Root Cause

In `tokenizers.py` lines 438-446:

```python
if len(exprs) > 0 and isinstance(exprs[0], int | np.integer):
    # Already binned by window function - just convert to tokens
    expr_tokens = np.array(exprs, dtype=np.int64) + self.expr_bin_start  # NO CLIPPING!
else:
    # Raw values - need to bin them
    expr_tokens = self._expression_to_bin_vectorized(...)  # This clips correctly
```

The issue: If raw expression values in SLAF data are stored as **integers** (counts like 148) rather than floats, and `use_binned_expressions=False`:

1. Window returns raw integer counts (e.g., 148)
2. Tokenizer sees integers, **assumes** they're pre-binned by window function
3. Adds `expr_bin_start` **without clipping**: `vocab_size + 148`
4. This exceeds `total_vocab_size = vocab_size + n_expression_bins`

## Symptoms

- CUDA assertion failure: `t >= 0 && t < n_classes`
- Token IDs exceed embedding vocabulary size
- Error occurs in `F.cross_entropy` for expression loss

## Example

With config:
- `vocab_size = 62714`
- `n_expression_bins = 100`
- `total_vocab_size = 62814`

Observed token: `62862` (bin_id = 148, exceeds n_expression_bins)

## Proposed Fix

Add clipping in the integer branch of tokenizers.py:

```python
if len(exprs) > 0 and isinstance(exprs[0], int | np.integer):
    # Clip to valid bin range in case values aren't pre-binned
    expr_bins = np.clip(
        np.array(exprs, dtype=np.int64),
        0,
        self.n_expression_bins - 1
    )
    expr_tokens = expr_bins + self.expr_bin_start
```

## Workaround

Set `n_expression_bins` high enough to cover max raw expression value (e.g., 200+).

## Files to Modify

- `slaf/slaf/ml/tokenizers.py` - Add clipping in integer expression branch

## Testing

1. Check SLAF data schema: `slaf_array.X.schema` to see if 'value' is int or float
2. Verify max expression value in dataset
3. After fix, verify expression tokens stay within `[vocab_size, total_vocab_size)`

## Related

- fast-scgpt train.py expression target calculation
- ScGPTWindow in aggregators.py (correctly clips when `use_binned_expressions=True`)
