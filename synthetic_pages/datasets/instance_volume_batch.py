from typing import NamedTuple
import numpy as np

class InstanceVolumeBatch(NamedTuple):
    vol: np.ndarray  # (H, W, D)
    lbl: np.ndarray  # (H, W, D)
