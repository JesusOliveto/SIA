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
    model = KMeansLoop(n_clusters=2, n_init=1, random_state=42, max_iter=50, tol=1e-4)
    model.fit(X)
    centers = np.sort(model.cluster_centers_, axis=0)
    assert centers.shape == (2, 2)
    assert np.allclose(centers[0], [0.33333333, 0.33333333], atol=1e-3)
    assert np.allclose(centers[1], [10.33333333, 10.33333333], atol=1e-3)


def test_numpy_matches_loop_inertia():
    X = _toy_data()
    loop = KMeansLoop(n_clusters=2, n_init=1, random_state=123)
    np_model = KMeansNumpy(n_clusters=2, n_init=1, random_state=123)
    loop.fit(X)
    np_model.fit(X)
    assert np.isclose(loop.inertia_, np_model.inertia_, rtol=1e-5)


def test_empty_cluster_handled():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    model = KMeansLoop(n_clusters=3, n_init=1, random_state=7, max_iter=20)
    model.fit(X)
    assert not np.isnan(model.cluster_centers_).any()
    assert len(np.unique(model.labels_)) == 3
