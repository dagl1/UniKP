#!/usr/bin/env python
"""
Validate Python 3.11 migration for UniKP infer_Kcats.
Run this to confirm the migration is working.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("default")


def check_python_version():
    """Verify Python 3.11+"""
    if sys.version_info < (3, 11):
        print(
            f"⚠ Warning: Python {sys.version_info.major}.{sys.version_info.minor} detected (3.11+ recommended)"
        )
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True


def check_compat_module():
    """Check compat_sklearn module exists and imports"""
    try:
        from compat_sklearn import safe_load_sklearn_model

        print("✓ compat_sklearn module loads")
        return True
    except Exception as e:
        print(f"✗ compat_sklearn import failed: {e}")
        return False


def check_infer_kcats_imports():
    """Check infer_Kcats can import (without running main)"""
    try:
        import infer_Kcats

        print("✓ infer_Kcats imports successfully")
        return True
    except Exception as e:
        print(f"✗ infer_Kcats import failed: {e}")
        return False


def check_sklearn_version():
    """Check sklearn version"""
    try:
        import sklearn

        version = sklearn.__version__
        print(f"✓ scikit-learn {version} installed")
        return True
    except Exception as e:
        print(f"✗ scikit-learn check failed: {e}")
        return False


def check_dependency_files():
    """Check that migration files exist"""
    required_files = [
        "compat_sklearn.py",
        "infer_Kcats.py",
        "requirements_python311.txt",
        "PYTHON_311_MIGRATION.md",
        "README_PYTHON311.md",
    ]

    all_ok = True
    for fname in required_files:
        fpath = Path(fname)
        if fpath.exists():
            print(f"✓ {fname} exists")
        else:
            print(f"✗ {fname} missing")
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print("UniKP Python 3.11 Migration Validation")
    print("=" * 60)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("scikit-learn", check_sklearn_version),
        ("Compatibility Module", check_compat_module),
        ("Dependency Files", check_dependency_files),
        ("infer_Kcats Imports", check_infer_kcats_imports),
    ]

    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Overall: {passed}/{total} checks passed")

    if passed == total:
        print("\n✅ Migration complete! You can now run:")
        print("   python infer_Kcats.py")
        return 0
    else:
        print("\n⚠ Some checks failed. Review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
