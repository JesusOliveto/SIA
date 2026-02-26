from __future__ import annotations

from typing import Protocol

import numpy as np


class KMeansLike(Protocol):
    """
    Protocolo o interfaz base que define los métodos y atributos requeridos
    para cualquier implementación del algoritmo K-Means en este proyecto.
    
    Finalidad:
    Garantizar que todas las implementaciones (Loop, NumPy, Scikit-learn)
    sean intercambiables y respeten la misma API pública, facilitando su uso
    polimórfico en el sistema de evaluación y visualización.
    """
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
    """
    Calcula la inercia (Suma de los Cuadrados dentro del Cluster, WCSS).

    ¿Qué hace?:
    Mide cuán internamente compactos son los clusters. Calcula la suma de la las
    distancias al cuadrado de cada muestra a su centroide asignado.

    ¿Cómo lo hace?:
    Resta a cada punto de datos 'X' las coordenadas de su centroide
    correspondiente (centers[labels]) y realiza la suma global de sus cuadrados.

    Finalidad:
    La inercia es la métrica intrínseca principal que el algoritmo K-Means minimiza.
    Se utiliza para evaluar la calidad del clustering (útil para el método del codo)
    y para seleccionar la mejor ejecución entre múltiples intentos (n_init).
    """
    diffs = X - centers[labels]
    return float(np.sum(diffs * diffs))
