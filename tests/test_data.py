from pathlib import Path

import numpy as np

from src.data import ZScoreScaler, load_winequality


def test_zscore_scaler_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    scaler = ZScoreScaler()
    Xn = scaler.fit_transform(X)
    assert np.allclose(Xn.mean(axis=0), [0.0, 0.0], atol=1e-7)
    assert np.allclose(Xn.std(axis=0), [1.0, 1.0], atol=1e-7)


def test_load_winequality_smoke():
    path = Path(__file__).resolve().parents[1] / "winequality.arff"
    bundle = load_winequality(path)
    assert bundle.X.shape[0] > 0
    assert bundle.X.shape[1] == len(bundle.feature_names)
