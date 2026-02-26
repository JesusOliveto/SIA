from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class KMeansSklearn:
    """
    Clase envoltorio (Thin wrapper) alrededor de la implementación KMeans de scikit-learn.
    
    ¿Qué hace?:
    Instancia y ejecuta el algoritmo K-Means provisto por la librería externa Scikit-Learn,
    adaptándolo a la interfaz `ModeloKMeans` del proyecto.
    
    ¿Cómo lo hace?:
    Delega internamente las llamadas `ajustar`, `predecir`, etc., a una instancia 
    del objeto `sklearn.cluster.KMeans` configurado con los parámetros provistos.
    
    Finalidad:
    Servir como marco de referencia "Gold Standard" (Patrón Oro). Permite comparar
    métricas de exactitud y tiempos de rendimiento o ejecución relativos a un motor
    comercial altamente optimizado escrito en C/Cython.
    """

    def __init__(self, num_clusters: int, max_iteraciones: int = 300, tolerancia: float = 1e-4, num_inicios: int = 10, estado_aleatorio: int | None = None) -> None:
        self.modelo = KMeans(
            n_clusters=num_clusters,
            max_iter=max_iteraciones,
            tol=tolerancia,
            n_init=num_inicios,
            random_state=estado_aleatorio,
            init="k-means++",
        )

    def ajustar(self, datos: np.ndarray) -> "KMeansSklearn":
        """
        Ajusta el modelo K-Means utilizando scikit-learn.
        
        Finalidad: Ejecuta la lógica optimizada de ajuste para el dataset y
        extrae los atributos resultantes al mismo formato del proyecto.
        """
        self.modelo.fit(datos)
        self.centroides_ = self.modelo.cluster_centers_
        self.etiquetas_ = self.modelo.labels_
        self.inercia_ = float(self.modelo.inertia_)
        self.num_iteraciones_ = int(self.modelo.n_iter_)
        return self

    def predecir(self, datos: np.ndarray) -> np.ndarray:
        return self.modelo.predict(datos)

    def ajustar_predecir(self, datos: np.ndarray) -> np.ndarray:
        return self.modelo.fit_predict(datos)
