"""
Compatibility utilities for loading legacy sklearn pickled models.

Handles the transition from sklearn 0.24.2 → 1.3+
where 'missing_go_to_left' was added to the internal tree-node dtype.
Our custom unpickler injects that field (defaulting to 1 / go-left) at
pickle-load time, *before* sklearn's __setstate__ validates the dtype.
"""

import io
import pickle
import warnings
from pathlib import Path
from pickle import BUILD, _Unpickler  # pure-Python implementation
from typing import Any

import numpy as np

# Integer opcode used as key in _Unpickler.dispatch
_BUILD_OPCODE: int = BUILD[0]


class _SklearnCompatUnpickler(_Unpickler):
    """
    Custom Unpickler that patches sklearn Tree node arrays during loading.

    When the BUILD opcode fires for a ``sklearn.tree._tree.Tree`` instance,
    the state dict (still on top of the stack at that point) is inspected.
    If the ``nodes`` array is missing the ``missing_go_to_left`` field
    (sklearn < 1.3 format), a new array with the expected dtype is created
    and the missing field is filled with 1 (samples with missing values go
    to the left child — the conservative pre-NaN-support default).
    """

    # Give this subclass its own dispatch table so the parent is not mutated.
    dispatch = dict(_Unpickler.dispatch)

    def load_build(self) -> None:  # type: ignore[override]
        stack = self.stack
        # At BUILD time the stack looks like: [..., inst, state]
        # super().load_build() will pop *state* and call inst.__setstate__(state).
        # We patch *before* the pop.
        if len(stack) >= 2:
            state = stack[-1]
            inst = stack[-2]
            if type(inst).__name__ == "Tree" and isinstance(state, dict) and "nodes" in state:
                self._maybe_fix_node_array(stack, state)
        super().load_build()

    def _maybe_fix_node_array(self, stack: list, state: dict) -> None:
        node_ndarray: np.ndarray = state["nodes"]
        if not (
            isinstance(node_ndarray, np.ndarray)
            and node_ndarray.dtype.names is not None
            and "missing_go_to_left" not in node_ndarray.dtype.names
        ):
            return  # already up-to-date, nothing to do

        try:
            from sklearn.tree._tree import NODE_DTYPE  # type: ignore[import]

            new_nodes = np.zeros(len(node_ndarray), dtype=NODE_DTYPE)
            for field in node_ndarray.dtype.names:
                new_nodes[field] = node_ndarray[field]
            # 1 → missing values routed to the left child (pre-1.3 behaviour)
            new_nodes["missing_go_to_left"] = 1
            # Replace state on stack with a patched copy
            stack[-1] = {**state, "nodes": new_nodes}
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not patch tree node array: {exc}. Predictions may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

    dispatch[_BUILD_OPCODE] = load_build  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Post-load attribute fixup
# ---------------------------------------------------------------------------


def _patch_sklearn_attributes(obj: Any) -> Any:
    """
    Walk a loaded sklearn estimator and fix attribute renames introduced
    between sklearn 0.24.x and 1.x+.

    Known renames:
    - ``base_estimator`` → ``estimator``  (renamed in 1.2, removed in 1.4+)
    """
    visited: set[int] = set()

    def _fix(o: Any) -> None:
        obj_id = id(o)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # base_estimator → estimator
        if hasattr(o, "base_estimator") and not hasattr(o, "estimator"):
            try:
                object.__setattr__(o, "estimator", o.base_estimator)
            except (AttributeError, TypeError):
                try:
                    o.__dict__["estimator"] = o.base_estimator
                except Exception:  # noqa: BLE001
                    pass

        # Recurse into child estimators (ensemble models)
        for attr in ("estimators_", "estimator", "base_estimator"):
            child = getattr(o, attr, None)
            if child is None:
                continue
            if isinstance(child, list):
                for c in child:
                    _fix(c)
            else:
                _fix(child)

    _fix(obj)
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def safe_load_sklearn_model(model_path: Path) -> Any:
    """
    Load a scikit-learn model pickle with compatibility for version mismatches.

    Handles ExtraTreeRegressor / decision-tree node-array dtype changes
    introduced in sklearn 1.3 (``missing_go_to_left`` field added).

    Strategy:
    1. Try a plain ``pickle.load`` (fast path for matching sklearn versions).
    2. On dtype mismatch, re-load via ``_SklearnCompatUnpickler`` which
       injects the missing field at the pickle protocol level — before
       sklearn's ``__setstate__`` can reject the old-format array.

    Args:
        model_path: Path to the pickled sklearn model.

    Returns:
        The loaded model.

    Raises:
        ValueError: If the model cannot be loaded even with the shim.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return _patch_sklearn_attributes(model)
    except (ValueError, TypeError) as first_err:
        if "node array from the pickle has an incompatible dtype" not in str(first_err):
            raise  # unrelated error — re-raise immediately

    warnings.warn(
        f"Detected sklearn version mismatch when loading '{model_path.name}'. "
        "Applying compatibility shim for missing 'missing_go_to_left' field "
        "(sklearn 0.24.x pickle loaded with sklearn 1.3+). "
        "For long-term reliability, re-export the model with your current sklearn.",
        UserWarning,
        stacklevel=2,
    )
    try:
        with open(model_path, "rb") as f:
            data = f.read()
        model = _SklearnCompatUnpickler(io.BytesIO(data)).load()
        return _patch_sklearn_attributes(model)
    except Exception as err:  # noqa: BLE001
        raise ValueError(
            f"Could not load '{model_path.name}' even with compatibility shim: {err}"
        ) from err
