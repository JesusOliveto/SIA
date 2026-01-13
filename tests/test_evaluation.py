import numpy as np

from src.evaluation import evaluate_models
from src.kmeans_loop import KMeansLoop
from src.kmeans_numpy import KMeansNumpy


def _toy():
    return np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.2, 9.8],
            [-10.0, -10.0],
            [-10.2, -9.8],
        ],
        dtype=np.float64,
    )


def test_evaluate_models_returns_results():
    X = _toy()
    builders = {
        "loop": lambda k, seed: KMeansLoop(n_clusters=k, random_state=seed, n_init=2, max_iter=50),
    }
    ks = [2, 3]
    results = evaluate_models(X, ks=ks, builders=builders, n_runs=2, random_state=0, compute_silhouette=True)
    assert len(results) == len(ks) * 2 * len(builders)
    for r in results:
        assert r.k in ks
        assert r.inertia >= 0.0
        if r.k > 1:
            assert r.silhouette is None or -1.0 <= r.silhouette <= 1.0


def test_multiple_impls_produce_outputs():
    X = _toy()
    builders = {
        "loop": lambda k, seed: KMeansLoop(n_clusters=k, random_state=seed, n_init=1, max_iter=30),
        "numpy": lambda k, seed: KMeansNumpy(n_clusters=k, random_state=seed, n_init=1, max_iter=30),
    }
    results = evaluate_models(X, ks=[2], builders=builders, n_runs=1, random_state=1)
    impls = {r.impl for r in results}
    assert impls == set(builders.keys())
