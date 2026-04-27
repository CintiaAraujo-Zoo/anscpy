"""
Tests for anscpy.gas._models

Each test verifies a mathematical property that must always be true,
regardless of the parameter values used.
"""

import numpy as np
import pytest
from anscpy.gas._models import (
    _model_le0,
    _model_lel,
    _model_mm,
    _model_mit,
    _model_expl,
    _model_gom,
    _model_log,
    _get_model_config,
)

TIME = np.array([0, 2, 4, 6, 8, 12, 24, 48, 72, 96], dtype=float)


# ---------------------------------------------------------------------------
# Property 1: all models must return 0 at t=0
# ---------------------------------------------------------------------------

def test_le0_zero_at_t0():
    assert _model_le0(0.0, Vf=150, k=0.06, d=1.0) == pytest.approx(0.0, abs=1e-6)

def test_lel_zero_at_t0():
    assert _model_lel(0.0, Vf=150, k=0.06, d=1.0, lag=2.0) == pytest.approx(0.0, abs=1e-6)

def test_mm_zero_at_t0():
    assert _model_mm(0.0, Vf=150, t_half=12, n=1.5) == pytest.approx(0.0, abs=1e-6)

def test_mit_zero_at_t0():
    assert _model_mit(0.0, Vf=150, b=0.03, c=0.10) == pytest.approx(0.0, abs=1e-6)

def test_expl_zero_at_t0():
    assert _model_expl(0.0, Vf=150, k=0.06, lag=2.0) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Property 2: all models must be monotonically increasing over time
# ---------------------------------------------------------------------------

def test_le0_increasing():
    values = _model_le0(TIME, Vf=150, k=0.06, d=1.0)
    assert np.all(np.diff(values) >= 0)

def test_lel_increasing():
    values = _model_lel(TIME, Vf=150, k=0.06, d=1.0, lag=2.0)
    assert np.all(np.diff(values) >= 0)

def test_mm_increasing():
    values = _model_mm(TIME, Vf=150, t_half=12, n=1.5)
    assert np.all(np.diff(values) >= 0)

def test_mit_increasing():
    values = _model_mit(TIME, Vf=150, b=0.03, c=0.10)
    assert np.all(np.diff(values) >= 0)

def test_expl_increasing():
    values = _model_expl(TIME, Vf=150, k=0.06, lag=2.0)
    assert np.all(np.diff(values) >= 0)

def test_gom_increasing():
    values = _model_gom(TIME, Vf=150, b=2.0, c=0.10)
    assert np.all(np.diff(values) >= 0)

def test_log_increasing():
    values = _model_log(TIME, Vf=150, b=2.0, c=0.10)
    assert np.all(np.diff(values) >= 0)


# ---------------------------------------------------------------------------
# Property 3: all models must approach Vf at large t
# ---------------------------------------------------------------------------

T_LARGE = np.array([500, 1000, 5000], dtype=float)

def test_le0_approaches_vf():
    values = _model_le0(T_LARGE, Vf=150, k=0.06, d=1.0)
    assert np.all(values > 148.0)

def test_mm_approaches_vf():
    values = _model_mm(T_LARGE, Vf=150, t_half=12, n=1.5)
    assert np.all(values > 148.0)

def test_expl_approaches_vf():
    values = _model_expl(T_LARGE, Vf=150, k=0.06, lag=2.0)
    assert np.all(values > 148.0)


# ---------------------------------------------------------------------------
# Property 4: model config registry
# ---------------------------------------------------------------------------

def test_get_model_config_valid():
    for model in ['LE0', 'LEL', 'MM', 'MIT', 'EXPL', 'GOM', 'LOG']:
        cfg = _get_model_config(model, v_max=150, t_max=96)
        assert 'func' in cfg
        assert 'param_names' in cfg
        assert 'bounds' in cfg
        assert 'p0_grid' in cfg

def test_get_model_config_invalid():
    with pytest.raises(ValueError, match="not recognized"):
        _get_model_config('INVALID', v_max=150, t_max=96)

def test_get_model_config_case_insensitive():
    cfg = _get_model_config('le0', v_max=150, t_max=96)
    assert cfg['func'] == _model_le0
