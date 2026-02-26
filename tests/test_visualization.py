"""Tests for the new visualization helper functions."""
import numpy as np
import pytest

from src.visualization import graficar_dispersion_2d, graficar_dispersion_3d


def _labels(n: int, k: int) -> np.ndarray:
    """Create evenly distributed labels for n points and k clusters."""
    return np.array([i % k for i in range(n)], dtype=np.int32)


def test_plot_scatter_2d_returns_figure():
    X = np.random.default_rng(0).random((30, 2))
    labels = _labels(30, 3)
    fig = graficar_dispersion_2d(X, labels, ["feat_a", "feat_b"])
    assert fig is not None
    assert fig.layout.title.text is not None


def test_plot_scatter_2d_uses_feature_names_as_axes():
    X = np.random.default_rng(1).random((20, 2))
    labels = _labels(20, 2)
    fig = graficar_dispersion_2d(X, labels, ["alcohol", "pH"])
    # Plotly stores axis labels in layout
    assert "alcohol" in fig.layout.xaxis.title.text
    assert "pH" in fig.layout.yaxis.title.text


def test_plot_scatter_3d_returns_figure():
    X = np.random.default_rng(2).random((30, 3))
    labels = _labels(30, 3)
    fig = graficar_dispersion_3d(X, labels, ["a", "b", "c"])
    assert fig is not None
    assert fig.layout.title.text is not None


def test_plot_scatter_3d_has_three_traces_or_more():
    """Each cluster should produce at least one trace."""
    X = np.random.default_rng(3).random((30, 3))
    labels = _labels(30, 3)
    fig = graficar_dispersion_3d(X, labels, ["x1", "x2", "x3"])
    assert len(fig.data) >= 3
