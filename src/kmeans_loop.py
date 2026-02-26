from __future__ import annotations

import numpy as np

from .kmeans_base import compute_inertia
from .utils import ensure_rng


class KMeansLoop:
    """
    Implementación de K-Means utilizando bucles explícitos de Python (sin vectorizar).

    ¿Qué hace?:
    Agrupa un conjunto de datos en 'n_clusters' particiones distintas basándose en la
    similitud (distancia Euclidiana) entre las muestras.

    ¿Cómo lo hace?:
    Itera muestra por muestra y centroide por centroide, calculando distancias con
    bucles 'for' anidados estándar de Python. Es `O(N*K*D)` en Python puro.

    Finalidad:
    Su único propósito es educativo y de depuración. Sirve para visualizar claramente
    la lógica granular y paso a paso del algoritmo (asignación y recálculo) sin el
    nivel de abstracción que introduce la vectorización matricial de NumPy.
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
        """
        Devuelve las coordenadas de los centros de los clusters.
        
        Finalidad: Proveer acceso seguro de solo lectura a los centroides 
        una vez que el algoritmo ha finalizado su entrenamiento.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.cluster_centers_

    @property
    def labels(self) -> np.ndarray:
        """Devuelve las etiquetas de cada punto."""
        if self.labels_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.labels_

    def fit(self, X: np.ndarray) -> "KMeansLoop":
        """
        Calcula el clustering K-means utilizando bucles explícitos.

        ¿Qué hace?:
        Ejecuta el ciclo de entrenamiento completo. Puede ejecutar el algoritmo múltiples 
        veces (n_init) con diferentes semillas iniciales.

        ¿Cómo lo hace?:
        Itera 'n_init' veces llamando a '_run_single'. Al final, retiene el modelo
        con la menor inercia encontrada de entre todas las ejecuciones para evitar mínimos locales.

        Finalidad:
        Encontrar los mejores centroides para un conjunto de datos dado y asignar
        las etiquetas óptimas a cada muestra, preparándolo para inferencias.

        Args:
            X: Datos estructurados de forma (n_samples, n_features).

        Returns:
            self: El estimador ajustado.
        """
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
        """
        Ejecuta una única iteración (corrida independiente) del algoritmo K-Means.
        
        ¿Qué hace?:
        Inicializa aleatoriamente los centroides muestreando puntos del dataset 
        y realiza el ciclo iterativo clásico de Lloyd: 
        1) Asignar muestras, 2) Recalcular centroides.
        
        ¿Cómo lo hace?:
        Repite el ciclo hasta que el desplazamiento ('shift') de los centroides 
        en una iteración sea menor que una tolerancia 'tol', o se alcance 'max_iter'. 
        Al usar bucles 'for', permite inyectar fácilmente código de logging (verbose).
        
        Finalidad:
        Llevar a cabo la lógica algorítmica iterativa fundamental para llegar
        a una convergencia desde un punto de partida (seed) específico.
        
        Args:
            X: Datos de entrada.
            seed: Semilla aleatoria para reproducibilidad en esta ejecución.
            
        Returns:
            Tupla de (centros, etiquetas, inercia, n_iter).
        """
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
            
            if self.verbose:
                # Calculate inertia for logging (expensive but useful for debug)
                current_inertia = compute_inertia(X, new_centers, labels)
                print(f"[Loop] Iter {iteration+1}: desplazamiento={shift:.6f}, inercia={current_inertia:.4f}")

            centers = new_centers
            if shift <= self.tol:
                if self.verbose:
                    print(f"[Loop] Convergencia en iter {iteration+1}")
                break

        inertia = compute_inertia(X, centers, labels)
        return centers, labels, inertia, iteration + 1

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Paso 1 del algoritmo de Lloyd: Asignación.
        Asigna cada muestra al centroide más cercano utilizando bucles explícitos.
        
        ¿Cómo lo hace?:
        Itera sobre todas las muestras de 'X', y para cada muestra evalúa la distancia
        cuadrada frente a cada centroide para determinar y retornar el índice más próximo.
        
        Finalidad:
        Mostrar de la forma más rudimentaria, pero clara para la enseñanza de Python, 
        cómo se deciden los clústeres iterando elemento a elemento (O(N*K*D)).
        """
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
        """
        Paso 2 del algoritmo de Lloyd: Actualización.
        Recalcula los centroides como el promedio ponderado de los puntos asignados.
        
        ¿Cómo lo hace?: Suma secuencialmente mediante índices explícitos y luego
        divide el total acumulado entre el conteo de muestras de cada clúster.

        Finalidad: Desplazar los centroides geométricamente hacia el centro de masa de
        su partición respectiva para minimizar iterativamente la inercia del sistema.
        Devuelve información sobre clústeres no asignados (vacíos) para corregirlos.
        """
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
        """
        Maneja la anomalía matemática de clusters vacíos (sin muestras asignadas).

        ¿Cómo lo hace?:
        Encuentra el punto 'X' que tiene la mayor distancia global con respecto a 
        su centroide actual, y reubica allí el centroide huérfano (vacío).
        
        Finalidad:
        Asegurar robustez empírica del algoritmo. Sin esta gestión la varianza inter-cluster
        caería sin justificación y un centroide dejaría de impactar la partición del espacio.
        """
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
        """
        Inferencia de cluster. Predice a qué partición pertenece cada nueva muestra.

        ¿Qué hace?:
        No altera el estado del modelo entrenado. Simplemente proyecta nuevos datos 
        y los clasifica usando el mapa de distancias a los centroides establecidos.

        Finalidad:
        Habilitar la puesta en producción del agrupador sobre streams de
        datos venideros tras haber validado el modelo K-Means actual.

        Args:
            X: Nuevos datos a predecir.

        Returns:
            Etiquetas de cluster.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        X = np.asarray(X, dtype=np.float64)
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Calcula los centros de los clusters y predice el índice del cluster para cada muestra."""
        return self.fit(X).labels_
