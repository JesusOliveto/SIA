import numpy as np

from src.kmeans_loop import KMeansLoop
from src.kmeans_numpy import KMeansNumpy


def _toy_data():
    return np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [10.0, 10.0], [10.0, 11.0], [11.0, 10.0]],
        dtype=np.float64,
    )


def test_loop_converges_two_clusters():
    X = _toy_data()
    model = KMeansLoop(num_clusters=2, num_inicios=1, estado_aleatorio=42, max_iteraciones=50, tolerancia=1e-4)
    model.ajustar(X)
    centers = np.sort(model.centroides_, axis=0)
    assert centers.shape == (2, 2)
    assert np.allclose(centers[0], [0.33333333, 0.33333333], atol=1e-3)
    assert np.allclose(centers[1], [10.33333333, 10.33333333], atol=1e-3)


def test_numpy_matches_loop_inertia():
    X = _toy_data()
    loop = KMeansLoop(num_clusters=2, num_inicios=1, estado_aleatorio=123)
    np_model = KMeansNumpy(num_clusters=2, num_inicios=1, estado_aleatorio=123)
    loop.ajustar(X)
    np_model.ajustar(X)
    assert np.isclose(loop.inercia_, np_model.inercia_, rtol=1e-5)


def test_empty_cluster_handled():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    model = KMeansLoop(num_clusters=3, num_inicios=1, estado_aleatorio=7, max_iteraciones=20)
    model.ajustar(X)
    assert not np.isnan(model.centroides_).any()
    assert len(np.unique(model.etiquetas_)) == 3
