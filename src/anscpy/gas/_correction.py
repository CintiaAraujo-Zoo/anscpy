"""
anscpy.gas._correction
======================
Blank correction for in vitro gas production data.

Supports any number of blank vials, any aggregation method,
and data in PSI or mL — compatible with semi-automatic
(pressure transducer) and fully automatic techniques.
"""

import warnings
import numpy as np
from typing import Union, Callable

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# PSI → mL calibration defaults
# Adjust to match your laboratory manometer calibration.
# ---------------------------------------------------------------------------
PSI_SLOPE     = 4.4392   # mL per PSI unit
PSI_INTERCEPT = 0.8943   # mL (intercept)


def correct_blank(time,
                  volume,
                  blank,
                  method: Union[str, Callable] = 'mean',
                  blank_unit: str = 'ml',
                  sample_unit: str = 'ml',
                  slope_psi: float = PSI_SLOPE,
                  intercept_psi: float = PSI_INTERCEPT,
                  clip_negative: bool = True,
                  warn_negative: bool = True):
    """
    Correct sample gas volume by subtracting blank vial production.

    Parameters
    ----------
    time : array-like
        Incubation time vector (h).

    volume : array-like
        Sample volumes in mL or PSI (see sample_unit).

    blank : array-like or DataFrame
        Blank vial data. Accepted formats:
        - 1D list or array : single blank vial.
        - 2D list or array : N vials × T timepoints.
        - DataFrame        : each column is one blank vial.

    method : str or callable, default 'mean'
        Aggregation method for multiple blank vials.
        Options: 'mean', 'median', 'min', 'max', or a custom callable.

    blank_unit : str, default 'ml'
        Unit of blank data: 'ml' or 'psi'.

    sample_unit : str, default 'ml'
        Unit of sample data: 'ml' or 'psi'.

    slope_psi : float, default 4.4392
        Slope of the PSI → mL calibration regression.

    intercept_psi : float, default 0.8943
        Intercept of the PSI → mL calibration regression.

    clip_negative : bool, default True
        If True, replace negative corrected values with 0.

    warn_negative : bool, default True
        If True, emit a warning when negative values are detected.

    Returns
    -------
    v_corrected : np.ndarray
        Blank-corrected volumes (mL).

    v_blank_aggregated : np.ndarray
        Aggregated blank series (mL).
    """
    t = np.asarray(time, dtype=float)

    v_sample = _to_ml(np.asarray(volume, dtype=float),
                      sample_unit, slope_psi, intercept_psi)

    blank_arr = _parse_blank(blank)

    blank_ml = np.apply_along_axis(
        lambda row: _to_ml(row, blank_unit, slope_psi, intercept_psi),
        axis=1, arr=blank_arr
    )

    if blank_ml.shape[1] != len(t):
        raise ValueError(
            f"Blank has {blank_ml.shape[1]} timepoints, but sample data "
            f"has {len(t)}. Ensure both vectors share the same time structure."
        )

    n_vials = blank_ml.shape[0]
    if n_vials == 1:
        v_blank_agg = blank_ml[0]
    else:
        if callable(method) and not isinstance(method, str):
            v_blank_agg = method(blank_ml)
        elif method == 'mean':
            v_blank_agg = np.mean(blank_ml, axis=0)
        elif method == 'median':
            v_blank_agg = np.median(blank_ml, axis=0)
        elif method == 'min':
            v_blank_agg = np.min(blank_ml, axis=0)
        elif method == 'max':
            v_blank_agg = np.max(blank_ml, axis=0)
        else:
            raise ValueError(
                f"Aggregation method '{method}' not recognized. "
                "Use 'mean', 'median', 'min', 'max', or a callable."
            )

    v_corr = v_sample - v_blank_agg

    neg_mask = v_corr < 0
    if neg_mask.any() and warn_negative:
        neg_times  = t[neg_mask]
        neg_values = v_corr[neg_mask]
        msg = (
            f"\n[anscpy.gas] WARNING — Blank correction produced "
            f"{neg_mask.sum()} negative value(s):\n"
        )
        for ti, vi in zip(neg_times, neg_values):
            msg += f"    t = {ti:.1f} h  →  V_corrected = {vi:.4f} mL\n"
        if clip_negative:
            msg += "  → Replaced with 0 (clip_negative=True).\n"
            msg += "  → If this occurs at many timepoints, review your blank data."
        else:
            msg += "  → Kept as-is (clip_negative=False). Interpret with caution."
        warnings.warn(msg, UserWarning, stacklevel=2)

    if clip_negative:
        v_corr = np.maximum(v_corr, 0.0)

    return v_corr, v_blank_agg


def _to_ml(arr: np.ndarray, unit: str,
           slope: float, intercept: float) -> np.ndarray:
    """Convert array from PSI to mL, or return as-is if already in mL."""
    if unit.lower() == 'psi':
        v = slope * arr + intercept
        if arr[0] == 0.0:
            v[0] = 0.0
        return v
    elif unit.lower() in ('ml', 'volume'):
        return arr.copy()
    else:
        raise ValueError(
            f"unit='{unit}' is invalid. Use 'ml' or 'psi'."
        )


def _parse_blank(blank) -> np.ndarray:
    """Parse any blank format into a 2D array (N_vials × T_timepoints)."""
    if _PANDAS_AVAILABLE:
        import pandas as pd
        if isinstance(blank, pd.DataFrame):
            return blank.values.T
        if isinstance(blank, pd.Series):
            return blank.values.reshape(1, -1)

    arr = np.asarray(blank, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] > arr.shape[1]:
            return arr.T
        return arr
    else:
        raise ValueError("blank must be 1D or 2D (N_vials × T_timepoints).")
