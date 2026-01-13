from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import silhouette_score

from .kmeans_base import KMeansLike


@dataclass
class RunResult:
    impl: str
    k: int
    run: int
    inertia: float
    fit_time: float
    n_iter: int
    labels: np.ndarray
    silhouette: Optional[float] = None


def evaluate_models(
    X: np.ndarray,
    ks: Iterable[int],
    builders: Dict[str, Callable[[int, Optional[int]], KMeansLike]],
    n_runs: int = 3,
    random_state: Optional[int] = None,
    compute_silhouette: bool = False,
) -> List[RunResult]:
    """Run multiple K selections and implementations, returning comparable metrics."""

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
                        sil_val = float(silhouette_score(X, model.labels_))
                    except Exception:
                        sil_val = None
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
                    )
                )
    return results
