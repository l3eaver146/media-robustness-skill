"""Frozen deterministic diamond grid; no media I/O."""
from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=6)
def grid_field(h, w, pitch, line_width, eps):
    y, x = np.ogrid[:h, :w]
    # Lines x+y=k*pitch and x-y=k*pitch form diamond cells.
    a = np.mod(x+y, pitch).astype(np.float32)
    b = np.mod(x-y, pitch).astype(np.float32)
    distance = np.minimum(np.minimum(a, pitch-a), np.minimum(b, pitch-b)) / math.sqrt(2)
    coverage = np.clip(line_width/2 + .5 - distance, 0, 1)
    # Dark mesh, gently compensated cell interiors: no broad moving patches.
    # This keeps the unrounded field's mean approximately zero.
    mean = float(coverage.mean())
    field = (mean-coverage) / max(1-mean, mean, 1e-6) * eps
    delta = np.rint(np.clip(field, -eps, eps)).astype(np.int16)
    delta.setflags(write=False)
    return delta

