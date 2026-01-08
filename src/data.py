from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff


@dataclass
class DataBundle:
    X: np.ndarray
    y: Optional[np.ndarray]
    feature_names: List[str]


class ZScoreScaler:
    """Minimal z-score scaler to reuse in predict/UI."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        self.mean_ = np.asarray(X, dtype=np.float64).mean(axis=0)
        self.scale_ = np.asarray(X, dtype=np.float64).std(axis=0)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def load_winequality(path: str | Path) -> DataBundle:
    path = Path(path)
    raw, _ = arff.loadarff(path)
    df = pd.DataFrame(raw)

    feature_names = [c for c in df.columns if c.lower() != "class"]
    X = df[feature_names].to_numpy(dtype=np.float32)

    y: Optional[np.ndarray] = None
    if "class" in df.columns:
        y_raw = df["class"]
        if y_raw.dtype.kind in {"S", "U", "O"}:
            y = y_raw.astype(str).to_numpy()
        else:
            y = y_raw.to_numpy()
        y = y.astype(np.int32, copy=False)

    return DataBundle(X=X, y=y, feature_names=feature_names)


def normalize_bundle(bundle: DataBundle, scaler: Optional[ZScoreScaler] = None) -> Tuple[DataBundle, ZScoreScaler]:
    sc = scaler or ZScoreScaler()
    X_norm = sc.fit_transform(bundle.X) if scaler is None else sc.transform(bundle.X)
    return DataBundle(X=X_norm.astype(np.float32), y=bundle.y, feature_names=bundle.feature_names), sc
