from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class KMeansSklearn:
    """Thin wrapper around scikit-learn KMeans for comparison."""

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, n_init: int = 10, random_state: int | None = None) -> None:
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            random_state=random_state,
            init="k-means++",
        )

    def fit(self, X: np.ndarray) -> "KMeansSklearn":
        self.model.fit(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.inertia_ = float(self.model.inertia_)
        self.n_iter_ = int(self.model.n_iter_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)
