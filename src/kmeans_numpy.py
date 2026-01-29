from __future__ import annotations

import numpy as np

from .kmeans_base import compute_inertia
from .utils import ensure_rng


class KMeansNumpy:
    """K-Means implementation using NumPy vectorization."""

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, n_init: int = 1, random_state: int | None = None, verbose: bool = False) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def fit(self, X: np.ndarray) -> "KMeansNumpy":
        X = np.asarray(X, dtype=np.float64)
        base_rng = ensure_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0

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
            distances = self._pairwise_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            new_centers, empty = self._recompute_centers(X, labels)
            if empty:
                self._fix_empty_clusters(X, new_centers, distances, labels)

            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            
            if self.verbose:
                # Re-compute inertia for logging if needed, or just log shift
                # Ideally we compute labels here again for inertia, but we have labels from previous step
                # Let's verify shift is enough for basic debug, but user asked for step-by-step
                print(f"[Numpy] Iter {iteration+1}: shift={shift:.6f}")

            centers = new_centers
            if shift <= self.tol:
                if self.verbose:
                    print(f"[Numpy] Converged at iter {iteration+1}")
                break

        inertia = compute_inertia(X, centers, labels)
        return centers, labels, inertia, iteration + 1

    def _pairwise_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _recompute_centers(self, X: np.ndarray, labels: np.ndarray):
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        counts = np.bincount(labels, minlength=self.n_clusters).astype(np.int32)
        for c_idx in range(self.n_clusters):
            if counts[c_idx] > 0:
                centers[c_idx] = X[labels == c_idx].mean(axis=0)
        empty = [c for c in range(self.n_clusters) if counts[c] == 0]
        return centers, empty

    def _fix_empty_clusters(self, X: np.ndarray, centers: np.ndarray, distances: np.ndarray, labels: np.ndarray) -> None:
        closest = distances[np.arange(distances.shape[0]), labels]
        farthest_idx = int(np.argmax(closest))
        empty = np.setdiff1d(np.arange(self.n_clusters), np.unique(labels))
        for c_idx in empty:
            centers[c_idx] = X[farthest_idx]
            labels[farthest_idx] = c_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=np.float64)
        distances = self._pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
