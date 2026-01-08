from __future__ import annotations

from typing import Protocol

import numpy as np


class KMeansLike(Protocol):
    n_clusters: int
    cluster_centers_: np.ndarray
    labels_: np.ndarray
    inertia_: float
    n_iter_: int

    def fit(self, X: np.ndarray) -> "KMeansLike":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        ...


def compute_inertia(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    diffs = X - centers[labels]
    return float(np.sum(diffs * diffs))
