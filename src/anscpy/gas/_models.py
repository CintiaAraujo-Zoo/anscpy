"""
anscpy.gas._models
==================
Mathematical model equations for in vitro gas production kinetics.

Each function receives a time array and model-specific parameters,
and returns the predicted cumulative gas volume (mL).

References
----------
Wang, M. et al. (2011). Animal Feed Science and Technology, 165, 137-150.
France, J. et al. (2000). British Journal of Nutrition, 83, 143-150.
Groot, J.C.J. et al. (1996). Journal of Animal Science, 74, 2985-2991.
Orskov, E.R. & McDonald, I. (1979). Journal of Agricultural Science, 92, 499-503.
Schofield, P. et al. (1994). Journal of the Science of Food and Agriculture, 65, 107-115.
"""

import numpy as np


def _model_le0(t, Vf, k, d):
    """Logistic-Exponential without lag — Wang et al. (2011). [DEFAULT]

    V(t) = Vf * (1 - exp(-k*t)) / (1 + exp(ln(1/d) - k*t))

    Parameters
    ----------
    t : array-like
        Incubation time (h).
    Vf : float
        Asymptotic gas volume (mL).
    k : float
        Rate constant (h⁻¹).
    d : float
        Shape parameter (dimensionless).
    """
    d = np.maximum(d, 1e-8)
    b = np.log(1.0 / d)
    num = 1.0 - np.exp(-k * t)
    den = 1.0 + np.exp(b - k * t)
    return Vf * num / den


def _model_lel(t, Vf, k, d, lag):
    """Logistic-Exponential with lag — Wang et al. (2011).

    V(t) = Vf * (1 - exp(-k*dt)) / (1 + exp(ln(1/d) - k*dt))
           where dt = max(t - lag, 0)

    Parameters
    ----------
    t : array-like
        Incubation time (h).
    Vf : float
        Asymptotic gas volume (mL).
    k : float
        Rate constant (h⁻¹).
    d : float
        Shape parameter (dimensionless).
    lag : float
        Lag time (h).
    """
    d = np.maximum(d, 1e-8)
    b = np.log(1.0 / d)
    dt = np.maximum(t - lag, 0.0)
    num = 1.0 - np.exp(-k * dt)
    den = 1.0 + np.exp(b - k * dt)
    return Vf * num / den


def _model_mm(t, Vf, t_half, n):
    """Modified Michaelis-Menten — Groot et al. (1996).

    V(t) = Vf * t^n / (t_half^n + t^n)

    Parameters
    ----------
    t : array-like
        Incubation time (h).
    Vf : float
        Asymptotic gas volume (mL).
    t_half : float
        Time to reach 50% of Vf (h).
    n : float
        Hill coefficient (dimensionless).
    """
    t_safe = np.maximum(t, 1e-10)
    tn = t_safe ** n
    return Vf * tn / (t_half ** n + tn)


def _model_mit(t, Vf, b, c):
    """Mitscherlich — France et al. (2000).

    V(t) = Vf * (1 - exp(-b*t - c*sqrt(t)))

    Parameters
    ----------
    t : array-like
        Incubation time (h).
    Vf : float
        Asymptotic gas volume (mL).
    b : float
        Rate parameter (h⁻¹).
    c : float
        Rate parameter (h⁻⁰·⁵).
    """
    return Vf * (1.0 - np.exp(-b * t - c * np.sqrt(t)))


def _model_expl(t, Vf, k, lag):
    """Exponential with lag — Orskov & McDonald (1979).

    V(t) = Vf * (1 - exp(-k * max(t - lag, 0)))

    Parameters
    ----------
    t : array-like
        Incubation time (h).
    Vf : float
        Asymptotic gas volume (mL).
    k : float
        Rate constant (h⁻¹).
    lag : float
        Lag time (h).
    """
    dt = np.maximum(t - lag, 0.0)
    return Vf * (1.0 - np.exp(-k * dt))


def _model_gom(t, Vf, b, c):
    """Gompertz — Schofield et al. (1994).

    V(t) = Vf * exp(-exp(b - c*t))

    Parameters
    ----------
    t : array-like
        Incubation time (h).
    Vf : float
        Asymptotic gas volume (mL).
    b : float
        Shape parameter (dimensionless).
    c : float
        Rate constant (h⁻¹).
    """
    return Vf * np.exp(-np.exp(b - c * t))


def _model_log(t, Vf, b, c):
    """Logistic — Schofield et al. (1994).

    V(t) = Vf / (1 + exp(b - c*t))

    Parameters
    ----------
    t : array-like
        Incubation time (h).
    Vf : float
        Asymptotic gas volume (mL).
    b : float
        Shape parameter (dimensionless).
    c : float
        Rate constant (h⁻¹).
    """
    return Vf / (1.0 + np.exp(b - c * t))


def _get_model_config(model_name: str, v_max: float, t_max: float) -> dict:
    """Return fitting configuration for a given model.

    Parameters
    ----------
    model_name : str
        Model identifier. One of: 'LE0', 'LEL', 'MM', 'MIT', 'EXPL', 'GOM', 'LOG'.
    v_max : float
        Maximum observed volume (mL). Used to set parameter bounds.
    t_max : float
        Maximum observed time (h). Used to set parameter bounds.

    Returns
    -------
    dict
        Dictionary with keys: func, n_params, param_names, bounds, p0_grid.

    Raises
    ------
    ValueError
        If model_name is not recognized.
    """
    configs = {
        'LE0': {
            'func': _model_le0, 'n_params': 3,
            'param_names': ['Vf (mL)', 'k (h⁻¹)', 'd'],
            'bounds': ([0, 0.001, 1e-6], [v_max * 3, 2.0, 100]),
            'p0_grid': [
                [v_max * f, k, d]
                for f in [0.8, 1.0, 1.2]
                for k in [0.02, 0.06, 0.12]
                for d in [0.3, 1.0, 3.0]
            ]
        },
        'LEL': {
            'func': _model_lel, 'n_params': 4,
            'param_names': ['Vf (mL)', 'k (h⁻¹)', 'd', 'lag (h)'],
            'bounds': ([0, 0.001, 1e-6, 0], [v_max * 3, 2.0, 100, t_max / 4]),
            'p0_grid': [
                [v_max * f, k, d, L]
                for f in [0.8, 1.0, 1.2]
                for k in [0.02, 0.06, 0.12]
                for d in [0.3, 1.0, 3.0]
                for L in [0.5, 2.0, 5.0]
            ]
        },
        'MM': {
            'func': _model_mm, 'n_params': 3,
            'param_names': ['Vf (mL)', 't_half (h)', 'n'],
            'bounds': ([0, 0.1, 0.1], [v_max * 3, t_max * 2, 20]),
            'p0_grid': [
                [v_max * f, t_h, n]
                for f in [0.9, 1.1, 1.3]
                for t_h in [6, 15, 30]
                for n in [0.5, 1.5, 3.0]
            ]
        },
        'MIT': {
            'func': _model_mit, 'n_params': 3,
            'param_names': ['Vf (mL)', 'b (h⁻¹)', 'c (h⁻⁰·⁵)'],
            'bounds': ([0, 0.0, 0.0], [v_max * 3, 1.0, 2.0]),
            'p0_grid': [
                [v_max * f, b, c]
                for f in [0.9, 1.1, 1.3]
                for b in [0.005, 0.03, 0.10]
                for c in [0.03, 0.10, 0.25]
            ]
        },
        'EXPL': {
            'func': _model_expl, 'n_params': 3,
            'param_names': ['Vf (mL)', 'k (h⁻¹)', 'lag (h)'],
            'bounds': ([0, 0.001, 0], [v_max * 3, 2.0, t_max / 4]),
            'p0_grid': [
                [v_max * f, k, L]
                for f in [0.9, 1.1, 1.3]
                for k in [0.02, 0.06, 0.15]
                for L in [0.5, 2.0, 5.0]
            ]
        },
        'GOM': {
            'func': _model_gom, 'n_params': 3,
            'param_names': ['Vf (mL)', 'b', 'c (h⁻¹)'],
            'bounds': ([0, -10, 0.001], [v_max * 3, 20, 2.0]),
            'p0_grid': [
                [v_max * f, b, c]
                for f in [0.9, 1.1, 1.3]
                for b in [0.5, 1.5, 3.0]
                for c in [0.03, 0.10, 0.25]
            ]
        },
        'LOG': {
            'func': _model_log, 'n_params': 3,
            'param_names': ['Vf (mL)', 'b', 'c (h⁻¹)'],
            'bounds': ([0, -10, 0.001], [v_max * 3, 20, 2.0]),
            'p0_grid': [
                [v_max * f, b, c]
                for f in [0.9, 1.1, 1.3]
                for b in [0.5, 1.5, 3.0]
                for c in [0.03, 0.10, 0.25]
            ]
        },
    }

    name = model_name.upper()
    if name not in configs:
        raise ValueError(
            f"Model '{model_name}' not recognized. "
            f"Available options: {list(configs.keys())}"
        )
    return configs[name]
