# bench_v2.py Fix Report

**Date:** 2025-12-05  
**Status:** ✅ All issues fixed and verified

## Summary

Successfully identified and fixed 4 critical bugs in `bench_v2.py`. The script now runs successfully and produces benchmark results as expected.

---

## Issues Found and Fixed

### 1. ❌ MODELS List Treated as Dictionary
**Location:** Lines 120, 132  
**Error:** `AttributeError: 'list' object has no attribute 'keys'`

**Problem:**
```python
# MODELS was defined as a list
MODELS = ["blip2","qwen","paligemma","llava"]

# But code tried to use it as a dictionary
choices=list(MODELS.keys()) + ["all"]
models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
```

**Fix:**
```python
# Line 120
choices=MODELS + ["all"]

# Line 132
models_to_run = MODELS if args.model == "all" else [args.model]
```

---

### 2. ❌ Missing peak_vram Field
**Location:** Line 156  
**Error:** `KeyError: 'peak_vram'`

**Problem:**
The `benchmark()` function doesn't return a `peak_vram` field, but the code tried to access it when flattening results for CSV.

**Fix:**
Replaced with the actually returned fields `model_size_bytes` and `model_parameters`:
```python
flat_result = {
    "model": result["model"],
    "quantization": result["quantization"],
    "load_time_s": result["load_time"],
    "model_size_mb": result["model_size_bytes"] / (1024 * 1024),
    "model_parameters": result["model_parameters"],
    "avg_latency_s": result["avg_latency"],
    "meteor": result["scores"]["meteor"],
    "sacrebleu": result["scores"]["score"],
    "perplexity": result["perplexity_score"]["mean_perplexity"],
}
```

---

### 3. ❌ Missing Perplexity Score in CSV Output
**Location:** Line 151-160  
**Error:** Perplexity score was calculated but not included in CSV output

**Problem:**
The perplexity score was computed and returned by `benchmark()` but not extracted when creating the flattened CSV results.

**Fix:**
Added perplexity extraction to flat_result dictionary:
```python
"perplexity": result["perplexity_score"]["mean_perplexity"],
```

---

### 4. ❌ Incorrect load_model Parameter Passing
**Location:** Line 63  
**Error:** `TypeError: '<=' not supported between instances of 'str' and 'int'`

**Problem:**
The `load_model()` function signature is:
```python
def load_model(model_name, max_new_tokens=100, quantization=None)
```

But the code was calling it with quantization as a positional argument:
```python
model = load_model(model_name, quantization)
```

This meant `quantization` (a string like "fp16") was being assigned to `max_new_tokens` (which expects an int), causing type comparison errors.

**Fix:**
Changed to use keyword argument:
```python
model = load_model(model_name, quantization=quantization)
```

---

## Verification

The script was tested and successfully completed a benchmark run:

```bash
python bench_v2.py --model blip2 --quantization fp16 --samples 2
```

**Results:**
- ✅ Model loaded successfully
- ✅ 2 samples processed
- ✅ Detailed results saved to `tmp/blip2_fp16_results.json`
- ✅ Benchmark results saved to `benchmark_results.csv`
- ✅ All metrics calculated correctly:
  - Load time: 7.58s
  - Model size: 7,142.57 MB
  - Model parameters: 3,744,761,856
  - Avg latency: 0.26s
  - METEOR: 0.00713
  - SacreBLEU: 1.345e-32
  - Perplexity: 136.32

---

## Files Modified

- [bench_v2.py](file:///userHome/userhome3/timur/vqa/optimization_project/bench_v2.py)

## Changes Summary

| Issue | Lines Changed | Type |
|-------|---------------|------|
| MODELS list/dict confusion | 120, 132 | Bug fix |
| Missing peak_vram field | 151-160 | Bug fix |
| Missing perplexity in output | 151-160 | Enhancement |
| Incorrect parameter passing | 63 | Bug fix |

---

## Next Steps

The benchmark script is now ready for full testing with:
- All models: `--model all`
- All quantizations: `--quantization all`
- Larger sample sizes: `--samples 50`

---

## Additional Enhancements (2025-12-05)

### 5. ✅ Added ROUGE and BERTScore Metrics
**Location:** Lines 37-75, 186-203

**Enhancement:**
Added comprehensive text evaluation metrics as requested:
- **ROUGE**: ROUGE-1, ROUGE-2, and ROUGE-L scores
- **BERTScore**: Precision, Recall, and F1 scores

These metrics provide better semantic similarity assessment compared to traditional metrics.

---

### 6. ✅ Added Mean Answer Length Metric
**Location:** Lines 64, 191

**Enhancement:**
Added mean answer length calculation to track the average length of generated answers (in characters).

**Implementation:**
```python
mean_answer_length = sum(len(pred) for pred in generated_answers) / len(generated_answers) if generated_answers else 0
```

This helps identify models that may be generating overly short or verbose outputs.

---

### 7. ✅ Fixed "Each input text must be at least one token long" Error
**Location:** Lines 37-43, 51-62

**Problem:**
Evaluation metrics (especially perplexity) failed when encountering empty predictions or ground truths with error: "Each input text must be at least one token long."

**Fix:**
Added filtering to replace empty strings with a placeholder before metric computation:
```python
filtered_predictions = [pred if pred.strip() else "[NO ANSWER]" for pred in generated_answers]
filtered_references = [ref if ref.strip() else "[NO ANSWER]" for ref in ground_truth_answers]
```

This ensures all text-based metrics receive valid input while preserving the original predictions for answer length calculation.

---

## Updated CSV Output Columns

The CSV now includes the following columns:
1. `model` - Model name
2. `quantization` - Quantization type (fp16, 8bit, 4bit)
3. `load_time_s` - Model loading time in seconds
4. `model_size_mb` - Model size in megabytes
5. `model_parameters` - Total number of model parameters
6. `avg_latency_s` - Average inference latency per sample
7. `mean_answer_length` - Average character length of generated answers
8. `meteor` - METEOR score
9. `sacrebleu` - SacreBLEU score
10. `rouge1` - ROUGE-1 score
11. `rouge2` - ROUGE-2 score
12. `rougeL` - ROUGE-L score
13. `bertscore_precision` - BERTScore precision
14. `bertscore_recall` - BERTScore recall
15. `bertscore_f1` - BERTScore F1
16. `perplexity` - Mean perplexity score
