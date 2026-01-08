from __future__ import annotations

import numpy as np

from .kmeans_base import compute_inertia
from .utils import ensure_rng


class KMeansLoop:
    """K-Means implementation using explicit Python loops."""

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, n_init: int = 1, random_state: int | None = None) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def fit(self, X: np.ndarray) -> "KMeansLoop":
        X = np.asarray(X, dtype=np.float64)
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0

        base_rng = ensure_rng(self.random_state)
        for _ in range(self.n_init):
            seed = int(base_rng.integers(0, 1_000_000_000))
            centers, labels, inertia, n_iter = self._run_single(X, seed)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = np.asarray(best_centers, dtype=np.float64)
        self.labels_ = np.asarray(best_labels, dtype=np.int32)
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)
        return self

    def _run_single(self, X: np.ndarray, seed: int):
        rng = ensure_rng(seed)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centers = X[indices].copy()

        labels = np.zeros(n_samples, dtype=np.int32)
        for iteration in range(self.max_iter):
            labels = self._assign_labels(X, centers)
            new_centers, empty = self._recompute_centers(X, labels)
            if empty:
                self._fix_empty_clusters(X, new_centers, labels, rng)

            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            if shift <= self.tol:
                break

        inertia = compute_inertia(X, centers, labels)
        return centers, labels, inertia, iteration + 1

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=np.int32)
        for i in range(n_samples):
            best_label = 0
            best_dist = np.inf
            for c_idx in range(self.n_clusters):
                dist = float(np.sum((X[i] - centers[c_idx]) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best_label = c_idx
            labels[i] = best_label
        return labels

    def _recompute_centers(self, X: np.ndarray, labels: np.ndarray):
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        counts = np.zeros(self.n_clusters, dtype=np.int32)
        for idx, label in enumerate(labels):
            centers[label] += X[idx]
            counts[label] += 1

        empty_clusters = []
        for c_idx in range(self.n_clusters):
            if counts[c_idx] == 0:
                empty_clusters.append(c_idx)
            else:
                centers[c_idx] /= counts[c_idx]
        return centers, empty_clusters

    def _fix_empty_clusters(self, X: np.ndarray, centers: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> None:
        distances = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            c_idx = labels[i]
            distances[i] = float(np.sum((X[i] - centers[c_idx]) ** 2))
        farthest_idx = int(np.argmax(distances))
        empty = np.setdiff1d(np.arange(self.n_clusters), np.unique(labels))
        for c_idx in empty:
            centers[c_idx] = X[farthest_idx]
            labels[farthest_idx] = c_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=np.float64)
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
