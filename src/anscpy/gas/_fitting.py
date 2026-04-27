"""
anscpy.gas._fitting
====================
Main fitting functions for in vitro gas production kinetics.

Public functions
----------------
fit_gas_production : Fit a model to a single sample.
fit_treatment      : Fit a model to multiple replicates of one treatment.
results_table      : Summarize a list of results into a DataFrame.
"""

import warnings
import numpy as np
from typing import Optional, Union, Callable
from scipy.optimize import curve_fit

from ._models import _get_model_config
from ._correction import correct_blank, _to_ml, PSI_SLOPE, PSI_INTERCEPT
from ._results import GasProductionResult

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def fit_gas_production(time,
                       volume,
                       input_unit: str = 'ml',
                       model: str = 'LE0',
                       blank=None,
                       blank_method: Union[str, Callable] = 'mean',
                       blank_unit: Optional[str] = None,
                       clip_negative: bool = True,
                       warn_negative: bool = True,
                       slope_psi: float = PSI_SLOPE,
                       intercept_psi: float = PSI_INTERCEPT,
                       p0=None,
                       treatment: str = '',
                       replicate: str = '',
                       verbose: bool = True):
    """
    Fit a mathematical model to in vitro gas production data.

    Parameters
    ----------
    time : array-like
        Incubation time vector (h).
        Example: [0, 2, 4, 6, 8, 12, 24, 48, 72, 96]

    volume : array-like
        Sample readings in PSI or mL, according to input_unit.

    input_unit : str, default 'ml'
        Unit of sample data: 'ml' or 'psi'.

    model : str, default 'LE0'
        Model to fit. Options: 'LE0', 'LEL', 'MM', 'MIT', 'EXPL', 'GOM', 'LOG'.

    blank : array-like or DataFrame, optional
        Blank vial data for correction.
        - None      : no correction applied.
        - 1D array  : single blank vial.
        - 2D array  : N blank vials × T timepoints.
        - DataFrame : each column is one blank vial.

    blank_method : str or callable, default 'mean'
        Aggregation method for multiple blank vials.
        Options: 'mean', 'median', 'min', 'max', or a custom callable.

    blank_unit : str, optional
        Unit of blank data. If None, inherits input_unit.

    clip_negative : bool, default True
        Replace negative corrected values with 0.

    warn_negative : bool, default True
        Emit a warning when negative corrected values are detected.

    slope_psi : float, default 4.4392
        Slope of the PSI → mL calibration regression.
        Adjust to match your laboratory manometer calibration.

    intercept_psi : float, default 0.8943
        Intercept of the PSI → mL calibration regression.

    p0 : list, optional
        Manual initial parameter values. If None, automatic grid search
        is used (recommended).

    treatment : str, optional
        Treatment name for result identification.

    replicate : str, optional
        Replicate identifier (e.g., 'R1', 'R2').

    verbose : bool, default True
        If True, print the fitting summary automatically.

    Returns
    -------
    GasProductionResult
        Object with estimated parameters, goodness-of-fit metrics,
        and .summary() / .plot() methods.

    Examples
    --------
    >>> from anscpy.gas import fit_gas_production
    >>> time   = [0, 2, 4, 6, 8, 12, 24, 48, 72, 96]
    >>> volume = [0, 10.3, 19.9, 28.9, 38.9, 61.9, 112.6, 140.1, 145.3, 149.1]
    >>> result = fit_gas_production(time, volume)
    >>> result.summary()
    >>> result.plot()
    """
    t   = np.asarray(time, dtype=float)
    raw = np.asarray(volume, dtype=float)

    if len(t) != len(raw):
        raise ValueError(
            f"'time' and 'volume' must have the same length, "
            f"but got {len(t)} and {len(raw)}."
        )

    v_raw = _to_ml(raw, input_unit, slope_psi, intercept_psi)

    v_blank_agg = None
    _blank_method_str = ""

    if blank is not None:
        _blank_unit = blank_unit if blank_unit is not None else input_unit
        _blank_method_str = (
            blank_method.__name__
            if callable(blank_method) and not isinstance(blank_method, str)
            else str(blank_method)
        )
        v_corr, v_blank_agg = correct_blank(
            t, raw,
            blank=blank,
            method=blank_method,
            blank_unit=_blank_unit,
            sample_unit=input_unit,
            slope_psi=slope_psi,
            intercept_psi=intercept_psi,
            clip_negative=clip_negative,
            warn_negative=warn_negative
        )
    else:
        v_corr = v_raw.copy()

    v_max = v_corr.max()
    if v_max <= 0:
        raise ValueError(
            "All corrected volumes are zero or negative. "
            "Check your blank data or sample readings."
        )

    cfg    = _get_model_config(model, v_max, t.max())
    func   = cfg['func']
    bounds = cfg['bounds']
    grid   = [p0] if p0 is not None else cfg['p0_grid']

    best_rmse = np.inf
    best_popt = None
    best_pcov = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p0_i in grid:
            try:
                popt, pcov = curve_fit(
                    func, t, v_corr,
                    p0=p0_i, bounds=bounds,
                    maxfev=50000, method='trf',
                    ftol=1e-10, xtol=1e-10
                )
                pred = func(t, *popt)
                rmse = np.sqrt(np.mean((pred - v_corr) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_popt = popt
                    best_pcov = pcov
            except Exception:
                continue

    if best_popt is None:
        raise RuntimeError(
            f"Model '{model}' did not converge for this dataset. "
            "Try providing manual initial values via p0=, or verify your data."
        )

    diag = np.diag(best_pcov)
    diag = np.where(diag < 0, 0.0, diag)
    se   = np.sqrt(diag)

    pred   = func(t, *best_popt)
    ss_res = np.sum((v_corr - pred) ** 2)
    ss_tot = np.sum((v_corr - np.mean(v_corr)) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n      = len(t)
    k_par  = cfg['n_params']
    aic    = n * np.log(ss_res / n) + 2 * k_par
    bic    = n * np.log(ss_res / n) + k_par * np.log(n)

    result = GasProductionResult(
        model_name   = model.upper(),
        param_names  = cfg['param_names'],
        popt         = best_popt,
        se           = se,
        r_squared    = r2,
        rmse         = best_rmse,
        aic          = aic,
        bic          = bic,
        n_obs        = n,
        time         = t,
        observed     = v_corr,
        observed_raw = v_raw,
        predicted    = pred,
        func         = func,
        blank_used   = v_blank_agg,
        blank_method = _blank_method_str,
        treatment    = treatment,
        replicate    = replicate,
        converged    = True
    )

    if verbose:
        result.summary()

    return result


def fit_treatment(time,
                  volume_matrix,
                  model: str = 'LE0',
                  input_unit: str = 'ml',
                  blank=None,
                  blank_method: Union[str, Callable] = 'mean',
                  blank_unit: Optional[str] = None,
                  treatment_name: str = '',
                  verbose: bool = False,
                  **kwargs):
    """
    Fit a model to multiple replicates of a single treatment.

    Parameters
    ----------
    time : array-like
        Time vector (h), shared across all replicates.

    volume_matrix : 2D array-like or DataFrame
        Volume data. Each column is one replicate.
        Example: DataFrame with columns ['R1', 'R2', 'R3', 'R4'].

    model : str, default 'LE0'
        Model to fit.

    input_unit : str, default 'ml'
        Unit of sample data: 'ml' or 'psi'.

    blank : array-like or DataFrame, optional
        Blank vial data applied to all replicates.

    blank_method : str or callable, default 'mean'
        Aggregation method for multiple blank vials.

    blank_unit : str, optional
        Unit of blank data. If None, inherits input_unit.

    treatment_name : str, optional
        Treatment name for result identification.

    verbose : bool, default False
        If True, print the summary for each replicate.

    Returns
    -------
    list of GasProductionResult
        One result object per replicate.
    """
    if _PANDAS_AVAILABLE and isinstance(volume_matrix, pd.DataFrame):
        col_names = list(volume_matrix.columns)
        mat = volume_matrix.values
    else:
        mat = np.asarray(volume_matrix, dtype=float)
        col_names = [f"R{i+1}" for i in range(mat.shape[1])]

    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)

    t = np.asarray(time, dtype=float)
    results = []

    for i in range(mat.shape[1]):
        rep_label = col_names[i]
        try:
            res = fit_gas_production(
                t, mat[:, i],
                model=model,
                input_unit=input_unit,
                blank=blank,
                blank_method=blank_method,
                blank_unit=blank_unit,
                treatment=treatment_name,
                replicate=rep_label,
                verbose=verbose,
                **kwargs
            )
            results.append(res)
        except Exception as e:
            warnings.warn(
                f"[anscpy.gas] Replicate '{rep_label}' failed: {e}",
                UserWarning
            )

    print(
        f"\n  Fitting complete: {len(results)}/{mat.shape[1]} "
        f"replicates converged — Treatment: {treatment_name or 'N/A'}"
    )

    return results


def results_table(results, include_se: bool = False):
    """
    Summarize a list of fitting results into a DataFrame.

    Parameters
    ----------
    results : list of GasProductionResult
        Results returned by fit_treatment() or a manual list.

    include_se : bool, default False
        If True, include standard error columns for each parameter.

    Returns
    -------
    pandas.DataFrame
        One row per replicate, with parameters and fit metrics.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for results_table(). "
            "Install it with: pip install anscpy[full]"
        )

    rows = []
    for r in results:
        row = {
            'ID':        f"{r.treatment} {r.replicate}".strip(),
            'Model':     r.model_name,
            'Blank':     'Yes' if r.blank_used is not None else 'No',
            'R²':        round(r.r_squared, 6),
            'RMSE (mL)': round(r.rmse, 4),
            'AIC':       round(r.aic, 2),
            'BIC':       round(r.bic, 2),
        }
        for i, name in enumerate(r.param_names):
            row[name] = round(r.popt[i], 4)
            if include_se:
                row[f'SE_{name}'] = round(r.se[i], 4)
        rows.append(row)

    return pd.DataFrame(rows)
