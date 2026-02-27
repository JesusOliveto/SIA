from pathlib import Path

import numpy as np

from src.data import EscaladorZScore, cargar_calidad_vino


def test_zscore_scaler_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    scaler = EscaladorZScore()
    Xn = scaler.ajustar_transformar(X)
    assert np.allclose(Xn.mean(axis=0), [0.0, 0.0], atol=1e-7)
    assert np.allclose(Xn.std(axis=0), [1.0, 1.0], atol=1e-7)


def test_load_winequality_smoke():
    path = Path(__file__).resolve().parents[1] / "datasets" / "winequality.arff"
    bundle = cargar_calidad_vino(path)
    assert bundle.datos.shape[0] > 0
    assert bundle.datos.shape[1] == len(bundle.nombres_caracteristicas)
