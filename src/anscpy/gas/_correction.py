"""
anscpy.gas._correction
======================
Blank correction for in vitro gas production data.

Input data must be in mL. Unit conversion (e.g., PSI → mL) is the
responsibility of the user and must be performed before calling any
function in this module.

Each laboratory should derive its own calibration equation experimentally.
See: Maurício et al. (1999), Pereira et al. (2009).
"""

import warnings
import numpy as np
from typing import Union, Callable

try:
    import pandas as pd  # noqa: F401
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def correct_blank(time,
                  volume,
                  blank,
                  method: Union[str, Callable] = 'mean',
                  clip_negative: bool = True,
                  warn_negative: bool = True):
    """
    Correct sample gas volume by subtracting blank vial production.

    All input data must be in mL. If your data is in PSI or another
    pressure unit, convert it to mL before calling this function using
    your laboratory's own calibration equation.

    Parameters
    ----------
    time : array-like
        Incubation time vector (h).

    volume : array-like
        Sample gas volumes in mL.

    blank : array-like or DataFrame
        Blank vial data in mL. Accepted formats:
        - 1D list or array : single blank vial.
        - 2D list or array : N vials × T timepoints.
        - DataFrame        : each column is one blank vial.

    method : str or callable, default 'mean'
        Aggregation method for multiple blank vials.
        Options: 'mean', 'median', 'min', 'max', or a custom callable.

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

    Examples
    --------
    >>> # Single blank vial
    >>> v_corr, v_blank = correct_blank(time, volume, blank=[0, 1.2, 2.3, ...])

    >>> # Two blank vials, median aggregation
    >>> v_corr, v_blank = correct_blank(
    ...     time, volume,
    ...     blank=[[0, 1.1, ...], [0, 1.3, ...]],
    ...     method='median'
    ... )

    >>> # DataFrame with N blank vials
    >>> v_corr, v_blank = correct_blank(time, volume, blank=df_blanks)
    """
    t        = np.asarray(time,   dtype=float)
    v_sample = np.asarray(volume, dtype=float)

    blank_arr = _parse_blank(blank)

    if blank_arr.shape[1] != len(t):
        raise ValueError(
            f"Blank has {blank_arr.shape[1]} timepoints, but sample data "
            f"has {len(t)}. Ensure both vectors share the same time structure."
        )

    n_vials = blank_arr.shape[0]
    if n_vials == 1:
        v_blank_agg = blank_arr[0]
    else:
        if callable(method) and not isinstance(method, str):
            v_blank_agg = method(blank_arr)
        elif method == 'mean':
            v_blank_agg = np.mean(blank_arr, axis=0)
        elif method == 'median':
            v_blank_agg = np.median(blank_arr, axis=0)
        elif method == 'min':
            v_blank_agg = np.min(blank_arr, axis=0)
        elif method == 'max':
            v_blank_agg = np.max(blank_arr, axis=0)
        else:
            raise ValueError(
                f"Aggregation method '{method}' not recognized. "
                "Use 'mean', 'median', 'min', 'max', or a callable."
            )

    v_corr   = v_sample - v_blank_agg
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
