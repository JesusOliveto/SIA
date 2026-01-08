from __future__ import annotations

from typing import Optional

import numpy as np


def ensure_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)
