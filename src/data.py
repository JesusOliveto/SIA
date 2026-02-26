from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff


@dataclass
class PaqueteDatos:
    """
    Estructura de datos centralizada (Data Transfer Object) para el proyecto.
    
    ¿Qué hace?:
    Encapsula la matriz de características (datos), las etiquetas verdaderas opcionales (etiquetas_reales),
    y los nombres legibles de cada columna (nombres_caracteristicas).
    
    Finalidad:
    Simplificar y estandarizar el paso de datos a través de las diferentes etapas 
    del pipeline interactivo (Carga -> Normalización -> Entrenamiento -> Evaluación -> Interfaz).
    """
    datos: np.ndarray
    etiquetas_reales: Optional[np.ndarray]
    nombres_caracteristicas: List[str]


class EscaladorZScore:
    """
    Normalizador estadístico Z-Score (Estandarización) minimalista.
    
    ¿Qué hace?:
    Transforma cada variable del dataset para tener media 0 y desviación estándar 1.
    Z = (X - Media) / Desviación_Estándar
    
    ¿Por qué usarlo en lugar de nada?:
    K-Means es un algoritmo basado estrictamente en distancias. Si las características
    están en escalas muy diferentes (ej: Alcohol en % vs Dióxido de Azufre en mg/L),
    las variables con valores numéricamente mayores dominarán la distancia asignando
    pesos artificiales que corrompen la topología geométrica real de los datos.
    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon
        self.media_: Optional[np.ndarray] = None
        self.escala_: Optional[np.ndarray] = None

    def ajustar(self, datos: np.ndarray) -> "EscaladorZScore":
        self.media_ = np.asarray(datos, dtype=np.float64).mean(axis=0)
        self.escala_ = np.asarray(datos, dtype=np.float64).std(axis=0)
        self.escala_ = np.where(self.escala_ < self.epsilon, 1.0, self.escala_)
        return self

    def transformar(self, datos: np.ndarray) -> np.ndarray:
        if self.media_ is None or self.escala_ is None:
            raise RuntimeError("Escalador no ajustado.")
        return (np.asarray(datos, dtype=np.float64) - self.media_) / self.escala_

    def ajustar_transformar(self, datos: np.ndarray) -> np.ndarray:
        return self.ajustar(datos).transformar(datos)


def cargar_calidad_vino(ruta: str | Path) -> PaqueteDatos:
    """
    Función de extracción y transformación inicial de datos (Extract & Load).
    
    ¿Qué hace?:
    Lee un archivo de datos tabulares en formato ARFF (estándar en repositorios 
    académicos como UCI), extrae los predictores numéricos y el ground_truth.
    
    Finalidad:
    Convertir datos estáticos serializados en disco en un formato consumible en memoria
    (matrices contiguas de NumPy) listo para cálculos matriciales C++.
    """
    ruta = Path(ruta)
    crudo, _ = arff.loadarff(ruta)
    df = pd.DataFrame(crudo)

    nombres_caracteristicas = [c for c in df.columns if c.lower() != "class"]
    datos_x = df[nombres_caracteristicas].to_numpy(dtype=np.float32)

    etiquetas_y: Optional[np.ndarray] = None
    if "class" in df.columns:
        y_crudo = df["class"]
        if y_crudo.dtype.kind in {"S", "U", "O"}:
            etiquetas_y = y_crudo.astype(str).to_numpy()
        else:
            etiquetas_y = y_crudo.to_numpy()
        etiquetas_y = etiquetas_y.astype(np.int32, copy=False)

    return PaqueteDatos(datos=datos_x, etiquetas_reales=etiquetas_y, nombres_caracteristicas=nombres_caracteristicas)


def normalizar_paquete(paquete: PaqueteDatos, escalador: Optional[EscaladorZScore] = None) -> Tuple[PaqueteDatos, EscaladorZScore]:
    """
    Aplica transformación a un PaqueteDatos existente. Devuelve un Paquete nuevo inmutable.
    Útil en Streamlit para no alterar el caché del dataset original.
    """
    esc = escalador or EscaladorZScore()
    datos_norm = esc.ajustar_transformar(paquete.datos) if escalador is None else esc.transformar(paquete.datos)
    return PaqueteDatos(datos=datos_norm.astype(np.float32), etiquetas_reales=paquete.etiquetas_reales, nombres_caracteristicas=paquete.nombres_caracteristicas), esc
