from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .kmeans_base import KMeansLike


@dataclass
class RunResult:
    """
    Estructura inmutable para registrar métricas de una sola iteracion (ejecución) de K-Means.
    
    ¿Qué hace?: 
    Contiene la metadata del modelo entrenado y los resultados multidimensionales,
    separando variables de estado (como labels), rendimiento e índices analíticos.
    """
    impl: str
    k: int
    run: int
    inertia: float
    fit_time: float
    n_iter: int
    labels: np.ndarray
    silhouette: Optional[float] = None
    ari: Optional[float] = None
    nmi: Optional[float] = None


def evaluate_models(
    X: np.ndarray,
    ks: Iterable[int],
    builders: Dict[str, Callable[[int, Optional[int]], KMeansLike]],
    n_runs: int = 3,
    random_state: Optional[int] = None,
    compute_silhouette: bool = False,
    y_true: Optional[np.ndarray] = None,
) -> List[RunResult]:
    """
    Orquestador del pipeline de evaluación para el análisis comparativo (Benchmark).

    ¿Qué hace?:
    Toma un conjunto de implementaciones (Loop, Numpy, Sklearn), un rango de `K` y las evalúa.

    ¿Cómo lo hace?:
    Itera exhaustivamente calculando en tiempo real:
    - Exactitud y penalización (Inercia).
    - Eficiencia algorítmica (Tiempo de ajuste en segundos `fit_time`).
    - Validación Interna: Silhouette Score si solicitado (cohesión y separación).
    - Validación Externa: (ARI y NMI) si se provee la clase real (y_true).

    Finalidad:
    Generar un log consolidado y agnóstico a la implementación (`RunResult`)
    que alimenta las visualizaciones académicas paramétricas de Streamlit.
    """

    rng = np.random.default_rng(random_state)
    results: List[RunResult] = []
    for k in ks:
        for run_idx in range(n_runs):
            seed = int(rng.integers(0, 1_000_000_000))
            for name, builder in builders.items():
                model = builder(k, seed)
                start = time.perf_counter()
                model.fit(X)
                elapsed = time.perf_counter() - start
                
                sil_val = None
                if compute_silhouette and k > 1:
                    try:
                        if len(np.unique(model.labels_)) > 1:
                            sil_val = float(silhouette_score(X, model.labels_))
                        else:
                            sil_val = -1.0
                    except Exception:
                        sil_val = None
                
                ari_val = None
                nmi_val = None
                if y_true is not None:
                    ari_val = float(adjusted_rand_score(y_true, model.labels_))
                    nmi_val = float(normalized_mutual_info_score(y_true, model.labels_))

                results.append(
                    RunResult(
                        impl=name,
                        k=k,
                        run=run_idx,
                        inertia=model.inertia_,
                        fit_time=elapsed,
                        n_iter=model.n_iter_,
                        labels=model.labels_,
                        silhouette=sil_val,
                        ari=ari_val,
                        nmi=nmi_val,
                    )
                )
    return results
