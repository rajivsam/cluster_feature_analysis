"""Row-wise invariance transformations."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class InvarianceTransformer(TransformerMixin, BaseEstimator):
    """Apply a domain-defined row-wise invariance.

    Parameters
    ----------
    invariance : {"none", "translation", "scale", "affine"}
        The row-wise transformation to apply.
    """

    _VALID_INVARIANCES = ("none", "translation", "scale", "affine")

    def __init__(self, invariance: str = "none") -> None:
        self.invariance = invariance

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self._validate_invariance()
        X_checked = check_array(X, dtype=float, ensure_2d=True)
        self.n_features_in_ = X_checked.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        X_checked = check_array(X, dtype=float, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )

        if self.invariance == "none":
            return X_checked.copy()
        if self.invariance == "translation":
            return X_checked - X_checked.mean(axis=1, keepdims=True)
        if self.invariance == "scale":
            norms = np.linalg.norm(X_checked, axis=1, keepdims=True)
            return X_checked / np.maximum(norms, 1e-12)

        centered = X_checked - X_checked.mean(axis=1, keepdims=True)
        standard_deviations = centered.std(axis=1, keepdims=True)
        return centered / np.maximum(standard_deviations, 1e-12)

    def _validate_invariance(self) -> None:
        if self.invariance not in self._VALID_INVARIANCES:
            choices = ", ".join(self._VALID_INVARIANCES)
            raise ValueError(
                f"Unknown invariance {self.invariance!r}; expected one of: "
                f"{choices}."
            )
