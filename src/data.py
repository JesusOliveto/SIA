from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff


@dataclass
class DataBundle:
    """
    Estructura de datos centralizada (Data Transfer Object) para el proyecto.
    
    ¿Qué hace?:
    Encapsula la matriz de características (X), las etiquetas verdaderas opcionales (y),
    y los nombres legibles de cada columna (feature_names).
    
    Finalidad:
    Simplificar y estandarizar el paso de datos a través de las diferentes etapas 
    del pipeline interactivo (Carga -> Normalización -> Entrenamiento -> Evaluación -> Interfaz).
    """
    X: np.ndarray
    y: Optional[np.ndarray]
    feature_names: List[str]


class ZScoreScaler:
    """
    Normalizador estadístico Z-Score (Estandarización) minimalista.
    
    ¿Qué hace?:
    Transforma cada variable del dataset para tener media 0 y desviación estándar 1.
    Z = (X - Media) / Desviación_Estándar
    
    ¿Por qué usarlo en lugar de nada?:
    K-Means es un algoritmo basado estrictamente en distancias. Si las características
    están en escalas muy diferentes (ej: Alcohol en % vs Dióxido de Azufre en mg/L),
    las variables con de valores numéricamente mayores dominarán la distancia asignando
    pesos artificiales que corrompen la topología geométrica real de los datos.
    """

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
    """
    Función de extracción y transformación inicial de datos (Extract & Load).
    
    ¿Qué hace?:
    Lee un archivo de datos tabulares en formato ARFF (estándar en repositorios 
    académicos como UCI), extrae los predictores numéricos y el ground_truth.
    
    Finalidad:
    Convertir datos estáticos serializados en disco en un formato consumible en memoria
    (matrices contiguas de NumPy) listo para cálculos matriciales C++.
    """
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
    """
    Aplica transformación a un DataBundle existente. Devuelve un Bundle nuevo inmutable.
    Útil en Streamlit para no alterar el caché del dataset original.
    """
    sc = scaler or ZScoreScaler()
    X_norm = sc.fit_transform(bundle.X) if scaler is None else sc.transform(bundle.X)
    return DataBundle(X=X_norm.astype(np.float32), y=bundle.y, feature_names=bundle.feature_names), sc
