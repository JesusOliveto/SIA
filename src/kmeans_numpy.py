from __future__ import annotations

import numpy as np

from .kmeans_base import compute_inertia
from .utils import ensure_rng


class KMeansNumpy:
    """
    Implementación de K-Means utilizando vectorización con NumPy.
    
    ¿Por qué esta implementación?
    ---------------------------
    Esta versión optimiza el cálculo utilizando las capacidades de broadcasting de NumPy.
    A diferencia de la implementación con bucles (`KMeansLoop`), esta versión realiza las
    operaciones matriciales en C (via NumPy), lo que la hace órdenes de magnitud más rápida
    y adecuada para datasets de tamaño real.
    
    Comparativa:
    - vs KMeansLoop: Mucho más rápida y eficiente en código, pero más opaca en la lógica paso a paso.
    - vs Sklearn: Implementación manual con fines educativos, sin las optimizaciones extremas (Cython) de Sklearn.
    """

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

    @property
    def cluster_centers(self) -> np.ndarray:
        """Devuelve las coordenadas de los centros de los clusters."""
        if self.cluster_centers_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.cluster_centers_

    @property
    def labels(self) -> np.ndarray:
        """Devuelve las etiquetas de cada punto."""
        if self.labels_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.labels_

    def fit(self, X: np.ndarray) -> "KMeansNumpy":
        """
        Calcula el clustering K-means.
        
        Esta función inicializa los centroides y realiza iteraciones hasta la convergencia.
        Utiliza operaciones vectorizadas de NumPy para el cálculo de distancias, lo que
        resulta en un rendimiento superior en comparación con la implementación basada en bucles.

        Args:
            X: Array-like de forma (n_samples, n_features).

        Returns:
            self: El estimador ajustado.
        """
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
        """
        Ejecuta una única iteración del algoritmo K-Means.
        
        Args:
            X: Datos de entrada.
            seed: Semilla aleatoria para esta ejecución.
            
        Returns:
            Tupla de (centros, etiquetas, inercia, n_iter).
        """
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
                print(f"[Numpy] Iter {iteration+1}: desplazamiento={shift:.6f}")

            centers = new_centers
            if shift <= self.tol:
                if self.verbose:
                    print(f"[Numpy] Convergencia en iter {iteration+1}")
                break

        inertia = compute_inertia(X, centers, labels)
        return centers, labels, inertia, iteration + 1

    def _pairwise_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Calcula las distancias Euclidianas al cuadrado utilizando broadcasting.
        
        Esta técnica aprovecha las operaciones C optimizadas de NumPy, evitando bucles lentos de Python.
        Es la clave para la eficiencia de esta implementación y la razón principal para usarla sobre
        `KMeansLoop` cuando se require rendimiento en datasets medianos/grandes.
        
        Args:
            X: Datos (N, D).
            centers: Centroides (K, D).
            
        Returns:
            Matriz de distancias (N, K).
        """
        diff = X[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _recompute_centers(self, X: np.ndarray, labels: np.ndarray):
        """Actualiza los centroides basándose en la media de los puntos asignados."""
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        counts = np.bincount(labels, minlength=self.n_clusters).astype(np.int32)
        for c_idx in range(self.n_clusters):
            if counts[c_idx] > 0:
                centers[c_idx] = X[labels == c_idx].mean(axis=0)
        empty = [c for c in range(self.n_clusters) if counts[c] == 0]
        return centers, empty

    def _fix_empty_clusters(self, X: np.ndarray, centers: np.ndarray, distances: np.ndarray, labels: np.ndarray) -> None:
        """
        Reinicializa clusters vacíos moviéndolos al punto más lejano.
        
        Estrategia: Encuentra el punto con la mayor distancia a su centroide actual
        y lo convierte en el nuevo centro para el cluster vacío.
        """
        closest = distances[np.arange(distances.shape[0]), labels]
        farthest_idx = int(np.argmax(closest))
        empty = np.setdiff1d(np.arange(self.n_clusters), np.unique(labels))
        for c_idx in empty:
            centers[c_idx] = X[farthest_idx]
            labels[farthest_idx] = c_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice el cluster más cercano al que pertenece cada muestra en X.

        Args:
            X: Nuevos datos a predecir.

        Returns:
            Índice del cluster al que pertenece cada muestra.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        X = np.asarray(X, dtype=np.float64)
        distances = self._pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Calcula los centros de los clusters y predice el índice del cluster para cada muestra."""
        return self.fit(X).labels_
