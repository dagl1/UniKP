#!/usr/bin/env python
"""Test script to verify compat_sklearn loads UniKP20kcat.pkl successfully."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("default")

try:
    from compat_sklearn import safe_load_sklearn_model

    print("Loading UniKP20kcat.pkl with compat_sklearn...")
    model_path = Path("UniKP20kcat.pkl")
    model = safe_load_sklearn_model(model_path)

    print("\n✓ SUCCESS: Model loaded successfully")
    print(f"  Type: {type(model).__name__}")
    print(f"  Has predict method: {hasattr(model, 'predict')}")

    # Try a small prediction to confirm it works
    import numpy as np

    test_input = np.random.randn(1, 512)
    try:
        prediction = model.predict(test_input)
        print(f"  Test prediction shape: {prediction.shape}")
        print(f"  Test prediction value: {prediction[0]}")
    except Exception as pred_err:
        print(f"  Warning: Prediction test failed: {pred_err}")

except Exception as e:
    print(f"\n✗ FAILED: {e}", file=sys.stderr)
    sys.exit(1)
