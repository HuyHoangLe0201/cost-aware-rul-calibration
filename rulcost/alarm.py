"""The alarm rule of the article: first time the smoothed point prediction falls to a threshold."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["first_alarm", "alarm_table"]


def first_alarm(predicted_rul, threshold: float, *, smoothing: int = 5, min_age: int = 10) -> Optional[Tuple[int, float]]:
    """Index and smoothed prediction of the first alarm in a trajectory of point RUL predictions.

    The prediction is smoothed by the median of the last ``smoothing`` values; the alarm is the first time index ``i >= min_age`` at which it is
    ``<= threshold``.  Returns ``None`` if the unit is never alarmed.
    """
    p = np.asarray(predicted_rul, dtype=float)
    smooth = np.array([np.median(p[max(0, i - smoothing + 1): i + 1]) for i in range(p.size)])
    hit = np.where((np.arange(p.size) >= min_age) & (smooth <= threshold))[0]
    if hit.size == 0:
        return None
    return int(hit[0]), float(smooth[hit[0]])


def alarm_table(predicted_trajectories: Sequence, true_rul_trajectories: Sequence, threshold: float, *, smoothing: int = 5, min_age: int = 10):
    """Apply :func:`first_alarm` to many run-to-failure units.

    Returns ``(unit_index, predicted_rul_at_alarm, true_rul_at_alarm)`` as arrays, for the units that were alarmed.
    """
    units, pred, true = [], [], []
    for u, (p, r) in enumerate(zip(predicted_trajectories, true_rul_trajectories)):
        a = first_alarm(p, threshold, smoothing=smoothing, min_age=min_age)
        if a is not None:
            units.append(u)
            pred.append(a[1])
            true.append(float(np.asarray(r, dtype=float)[a[0]]))
    return np.array(units), np.array(pred), np.array(true)
