from __future__ import annotations

import numpy as np

from .kmeans_base import compute_inertia
from .utils import ensure_rng


class KMeansNumpy:
    """
    Implementación de K-Means utilizando vectorización integral con NumPy.

    ¿Qué hace?:
    Agrupa un conjunto de datos particionándolo en 'n_clusters' utilizando la 
    metáfora probabilística del algoritmo iterativo original.

    ¿Cómo lo hace?:
    Reempleza los costosos iteradores `for` anidados de Python por tensores 
    y manipulación de ejes de memoria lineal usando "broadcasting". 

    Finalidad:
    Demostrar el impacto monumental del cálculo matricial delegando las instrucciones
    a C por debajo de la interfaz NumPy. Muestra un "punto intermedio" que es 
    suficientemente rápido para la vida real pero al que el desarrollador aún
    puede acceder para entender la implementación.
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
        Calcula el clustering K-means utilizando operaciones tensor-completas de NumPy.

        ¿Qué hace?:
        Coordina las iteraciones y los arranques ('n_init') del algoritmo buscando
        el mejor modelo posible (aquél con la métrica de inercia más baja).

        ¿Cómo lo hace?:
        Controla los reinicios y se apoya en `_run_single` para gestionar la convergencia
        vectorizada internamente. Garantiza conversiones estrictas a float64
        para evitar overflows de puntero flotante.
        
        Finalidad:
        Intermediario principal de la API para recibir un dataset completo, entrenar el
        modelo reteniendo las etiquetas/centroides idóneos, y devolver la propia instancia 
        ajustada (encadenamiento de métodos).
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
        Ejecuta una única corrida (ejecución independiente) del algoritmo K-Means.
        
        ¿Qué hace?:
        Lleva a cabo el proceso de inicialización y las fases de Lloyd: 
        1) Actualizar Asignaciones, 2) Actualizar Centroides.
        
        ¿Cómo lo hace?:
        Vectoriza brutalmente tanto la evaluación euclidiana como la recalculación.
        La condición de parada se computa determinando la norma matricial del 
        desplazamiento de clústeres (np.linalg.norm(new_centers - centers, axis=1)).
        
        Finalidad:
        Motor de procesamiento principal. Busca la convergencia en el menor tiempo 
        de cómputo posible evitando traspasos de memoria Python-runtime perjudiciales.
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
        Motor geométrico: Calcula distancias utilizando 'broadcasting'.
        
        ¿Qué hace?:
        Computa la distancia entre toda pareja de `(muestra, centroide)` posible 
        escalando dinámicamente las dimensiones de los tensores.
        
        ¿Cómo lo hace?:
        Proyecta X de formato (N, D) a (N, 1, D) y altera 'centers' a (1, K, D). 
        A partir de allí la resta se 'expande', generando un tensor 3D de 
        formato (N, K, D), para luego hacer sum() cuadrática sobre el eje D.
        
        Finalidad:
        Garantizar el punto más diferencial (O(1) ciclos for en Python) entre el 
        algoritmo base y la optimización NumPy. Devuelve la malla
        distancial necesaria para que argmin identifique los clusters.
        """
        diff = X[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _recompute_centers(self, X: np.ndarray, labels: np.ndarray):
        """
        Actualiza los centroides utilizando indexación booleana de NumPy.
        
        ¿Cómo lo hace?:
        Recuenta asignaciones con `np.bincount` de antemano. Itera sólo a través de los
        `K` clústeres para aplicar un filtrado booleano eficiente del tipo `X[labels == c_idx]`
        calculando su media de forma nativa en C con `.mean(axis=0)`.

        Finalidad:
        Mover cada clúster de forma ultra-rápida y concurrente a su verdadero centro de masa 
        sin el rastreo iterativo secuencial. Retorna además si un cluster se quedó sin datos.
        """
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        counts = np.bincount(labels, minlength=self.n_clusters).astype(np.int32)
        for c_idx in range(self.n_clusters):
            if counts[c_idx] > 0:
                centers[c_idx] = X[labels == c_idx].mean(axis=0)
        empty = [c for c in range(self.n_clusters) if counts[c] == 0]
        return centers, empty

    def _fix_empty_clusters(self, X: np.ndarray, centers: np.ndarray, distances: np.ndarray, labels: np.ndarray) -> None:
        """
        Mapea el algoritmo de salvaguarda contra "Singularidad" en operaciones tensoriales.
        
        ¿Cómo lo hace?:
        Usa indexaciones avanzadas como `distances[np.arange(...), labels]` para 
        acceder de forma O(1) a la matriz de desviaciones, buscando sistemáticamente
        el mayor outlier de todos a relocalizar y "salvar" el clúster inerte.

        Finalidad:
        Prevenir la formación de sub-clústeres vacíos donde `.mean(...)` retornaría
        Valores nulos, lo que corrompe la convergencia geométrica de la vectorización en curso.
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
