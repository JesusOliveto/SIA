from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .kmeans_base import ModeloKMeans


@dataclass
class ResultadoEjecucion:
    """
    Estructura inmutable para registrar métricas de una sola iteracion (ejecución) de K-Means.
    
    ¿Qué hace?: 
    Contiene la metadata del modelo entrenado y los resultados multidimensionales,
    separando variables de estado (como etiquetas), rendimiento e índices analíticos.
    """
    implementacion: str
    k: int
    corrida: int
    inercia: float
    tiempo_entrenamiento: float
    num_iteraciones: int
    etiquetas: np.ndarray
    silhouette: Optional[float] = None
    ari: Optional[float] = None
    nmi: Optional[float] = None


def evaluar_modelos(
    datos: np.ndarray,
    valores_k: Iterable[int],
    constructores: Dict[str, Callable[[int, Optional[int]], ModeloKMeans]],
    num_corridas: int = 3,
    estado_aleatorio: Optional[int] = None,
    calcular_silhouette: bool = False,
    etiquetas_reales: Optional[np.ndarray] = None,
) -> List[ResultadoEjecucion]:
    """
    Orquestador del pipeline de evaluación para el análisis comparativo (Benchmark).

    ¿Qué hace?:
    Toma un conjunto de implementaciones (Loop, Numpy, Sklearn), un rango de `K` y las evalúa.

    ¿Cómo lo hace?:
    Itera exhaustivamente calculando en tiempo real:
    - Exactitud y penalización (Inercia).
    - Eficiencia algorítmica (Tiempo de ajuste en segundos `tiempo_entrenamiento`).
    - Validación Interna: Silhouette Score si solicitado (cohesión y separación).
    - Validación Externa: (ARI y NMI) si se provee la clase real (etiquetas_reales).

    Finalidad:
    Generar un log consolidado y agnóstico a la implementación (`ResultadoEjecucion`)
    que alimenta las visualizaciones académicas paramétricas de Streamlit.
    """

    generador = np.random.default_rng(estado_aleatorio)
    resultados: List[ResultadoEjecucion] = []
    for k in valores_k:
        for idx_corrida in range(num_corridas):
            semilla = int(generador.integers(0, 1_000_000_000))
            for nombre, constructor in constructores.items():
                modelo = constructor(k, semilla)
                inicio = time.perf_counter()
                modelo.ajustar(datos)
                tiempo_transcurrido = time.perf_counter() - inicio
                
                valor_sil = None
                if calcular_silhouette and k > 1:
                    try:
                        if len(np.unique(modelo.etiquetas_)) > 1:
                            valor_sil = float(silhouette_score(datos, modelo.etiquetas_))
                        else:
                            valor_sil = -1.0
                    except Exception:
                        valor_sil = None
                
                valor_ari = None
                valor_nmi = None
                if etiquetas_reales is not None:
                    valor_ari = float(adjusted_rand_score(etiquetas_reales, modelo.etiquetas_))
                    valor_nmi = float(normalized_mutual_info_score(etiquetas_reales, modelo.etiquetas_))

                resultados.append(
                    ResultadoEjecucion(
                        implementacion=nombre,
                        k=k,
                        corrida=idx_corrida,
                        inercia=modelo.inercia_,
                        tiempo_entrenamiento=tiempo_transcurrido,
                        num_iteraciones=modelo.num_iteraciones_,
                        etiquetas=modelo.etiquetas_,
                        silhouette=valor_sil,
                        ari=valor_ari,
                        nmi=valor_nmi,
                    )
                )
    return resultados
