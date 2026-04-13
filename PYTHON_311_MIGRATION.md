# UniKP Python 3.11 Migration Guide

## Problem

`UniKP20kcat.pkl` was created with scikit-learn 0.24.2 (Python 3.8), which has an incompatible internal node array dtype in scikit-learn 1.8.0+ (Python 3.11).

**Error:**
```
ValueError: node array from the pickle has an incompatible dtype:
- expected: {'names': ['left_child', 'right_child', ..., 'missing_go_to_left'], ...}
- got     : [('left_child', '<i8'), ('right_child', '<i8'), ...]
```

## Solution

Use the new `compat_sklearn.py` module to safely load the legacy pickle with automatic compatibility handling. No need to maintain a separate Python 3.8 environment anymore.

## Setup

### Option A: Python 3.11 + Compatible Dependencies (Recommended)

```bash
cd UniKP
python -m venv venv_311
source venv_311/bin/activate  # or on Windows: venv_311\Scripts\activate

# Install dependencies; scikit-learn 0.24.x supports Python 3.11
pip install -r requirements_python311.txt
```

### Option B: Python 3.11 with sklearn 1.8.0+ (Requires Retraining)

If you want to use the latest sklearn 1.8.0:
1. Load the model with `safe_load_sklearn_model()` (handles compatibility)
2. Re-train with the new sklearn version
3. Save with joblib instead of pickle

## Files

- `compat_sklearn.py` — Compatibility wrapper for loading old sklearn pickles
- `requirements_python311.txt` — Python 3.11 compatible package versions
- `infer_Kcats.py` — Updated to use `safe_load_sklearn_model()`

## Running infer_Kcats.py

```bash
python infer_Kcats.py
```

The script will:
1. Attempt to load `UniKP20kcat.pkl` with standard pickle
2. If a dtype error is detected, automatically apply compatibility fixes
3. Warn if the model was loaded with potential dtype mismatches

## Long-Term Fix

To avoid this issue in the future:

1. **Use joblib instead of pickle** for sklearn models (joblib is sklearn's recommended persistence format):
   ```python
   from sklearn.externals import joblib
   joblib.dump(model, "model.joblib")
   model = joblib.load("model.joblib")
   ```

2. **Use sklearn.utils.estimator_html_repr** for model introspection without serialization.

3. **Keep requirements.txt pinned** to specific versions that work together.

