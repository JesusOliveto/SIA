import numpy as np

from src.evaluation import evaluar_modelos
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
        "loop": lambda k, seed: KMeansLoop(num_clusters=k, estado_aleatorio=seed, num_inicios=2, max_iteraciones=50),
    }
    ks = [2, 3]
    results = evaluar_modelos(X, valores_k=ks, constructores=builders, num_corridas=2, estado_aleatorio=0, calcular_silhouette=True)
    assert len(results) == len(ks) * 2 * len(builders)
    for r in results:
        assert r.k in ks
        assert r.inercia >= 0.0
        if r.k > 1:
            assert r.silhouette is None or -1.0 <= r.silhouette <= 1.0


def test_multiple_impls_produce_outputs():
    X = _toy()
    builders = {
        "loop": lambda k, seed: KMeansLoop(num_clusters=k, estado_aleatorio=seed, num_inicios=1, max_iteraciones=30),
        "numpy": lambda k, seed: KMeansNumpy(num_clusters=k, estado_aleatorio=seed, num_inicios=1, max_iteraciones=30),
    }
    results = evaluar_modelos(X, valores_k=[2], constructores=builders, num_corridas=1, estado_aleatorio=1)
    impls = {r.implementacion for r in results}
    assert impls == set(builders.keys())
