from __future__ import annotations

from typing import Optional

import numpy as np


def ensure_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Gestor de aleatoriedad centralizado del proyecto (Generador de Números Pseudoaleatorios).
    
    ¿Qué hace?:
    Instancia el nuevo motor recomendado de aleatoriedad de NumPy (PCG64) de 
    v1.17 en adelante en vez del histórico Mersenne Twister (np.random.seed).
    
    Finalidad:
    Garantizar la reproducibilidad estricta de todos los experimentos K-Means, 
    evaluaciones y benchmarks. Todo proceso estocástico del backend pedirá su
    `random_state` original derivando desde esta única semilla.
    """
    return np.random.default_rng(seed)
