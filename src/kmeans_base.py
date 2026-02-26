from __future__ import annotations

from typing import Protocol

import numpy as np


class ModeloKMeans(Protocol):
    """
    Protocolo o interfaz base que define los métodos y atributos requeridos
    para cualquier implementación del algoritmo K-Means en este proyecto.
    
    Finalidad:
    Garantizar que todas las implementaciones (Loop, NumPy, Scikit-learn)
    sean intercambiables y respeten la misma API pública, facilitando su uso
    polimórfico en el sistema de evaluación y visualización.
    """
    num_clusters: int
    centroides_: np.ndarray
    etiquetas_: np.ndarray
    inercia_: float
    num_iteraciones_: int

    def ajustar(self, datos: np.ndarray) -> "ModeloKMeans":
        ...

    def predecir(self, datos: np.ndarray) -> np.ndarray:
        ...

    def ajustar_predecir(self, datos: np.ndarray) -> np.ndarray:
        ...


def calcular_inercia(datos: np.ndarray, centroides: np.ndarray, etiquetas: np.ndarray) -> float:
    """
    Calcula la inercia (Suma de los Cuadrados dentro del Cluster, WCSS).

    ¿Qué hace?:
    Mide cuán internamente compactos son los clusters. Calcula la suma de las
    distancias al cuadrado de cada muestra a su centroide asignado.

    ¿Cómo lo hace?:
    Resta a cada punto de datos 'datos' las coordenadas de su centroide
    correspondiente (centroides[etiquetas]) y realiza la suma global de sus cuadrados.

    Finalidad:
    La inercia es la métrica intrínseca principal que el algoritmo K-Means minimiza.
    Se utiliza para evaluar la calidad del clustering (útil para el método del codo)
    y para seleccionar la mejor ejecución entre múltiples intentos (num_inicios).
    """
    diferencias = datos - centroides[etiquetas]
    return float(np.sum(diferencias * diferencias))
