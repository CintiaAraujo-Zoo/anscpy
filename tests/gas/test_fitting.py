"""
Tests for anscpy.gas._fitting

Verifies that fit_gas_production() converges correctly
using a reference dataset from Wang et al. (2011).

All input data is in mL. PSI conversion is the user's responsibility
and is not tested here.
"""

import pytest
import numpy as np
from anscpy.gas import fit_gas_production, fit_treatment, results_table

# ---------------------------------------------------------------------------
# Reference dataset — Wang et al. (2011)
# Used as ground truth for convergence and parameter tests
# ---------------------------------------------------------------------------

TIME   = [0, 2, 4, 6, 8, 12, 24, 48, 72, 96]
VOLUME = [0, 8.1, 16.3, 26.2, 38.4, 61.1, 115.2, 142.3, 148.1, 149.7]


# ---------------------------------------------------------------------------
# Basic convergence
# ---------------------------------------------------------------------------

def test_fit_returns_result():
    result = fit_gas_production(TIME, VOLUME, verbose=False)
    assert result is not None
    assert result.converged is True

def test_fit_r2_above_threshold():
    result = fit_gas_production(TIME, VOLUME, verbose=False)
    assert result.r_squared > 0.99

def test_fit_vf_reasonable():
    result = fit_gas_production(TIME, VOLUME, verbose=False)
    assert 140 < result.Vf < 165

def test_fit_rmse_reasonable():
    result = fit_gas_production(TIME, VOLUME, verbose=False)
    assert result.rmse < 5.0


# ---------------------------------------------------------------------------
# All models converge on the reference dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["LE0", "LEL", "MM", "MIT", "EXPL", "GOM", "LOG"])
def test_all_models_converge(model):
    result = fit_gas_production(TIME, VOLUME, model=model, verbose=False)
    assert result.r_squared > 0.95


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        fit_gas_production([0, 2, 4], [0, 10], verbose=False)

def test_invalid_model_raises():
    with pytest.raises(ValueError, match="not recognized"):
        fit_gas_production(TIME, VOLUME, model="INVALID", verbose=False)


# ---------------------------------------------------------------------------
# Blank correction — data in mL only
# ---------------------------------------------------------------------------

def test_fit_with_single_blank():
    blank = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.0, 5.2]
    result = fit_gas_production(
        TIME, VOLUME,
        blank=blank,
        verbose=False
    )
    assert result.blank_used is not None
    assert result.r_squared > 0.95

def test_fit_with_multiple_blanks():
    blank = [
        [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.0, 5.2],
        [0, 0.4, 0.9, 1.4, 1.9, 2.4, 3.3, 4.3, 4.8, 5.0],
    ]
    result = fit_gas_production(
        TIME, VOLUME,
        blank=blank,
        blank_method='mean',
        verbose=False
    )
    assert result.blank_used is not None
    assert result.r_squared > 0.95

def test_blank_result_stores_method():
    blank = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.0, 5.2]
    result = fit_gas_production(
        TIME, VOLUME,
        blank=blank,
        blank_method='median',
        verbose=False
    )
    assert result.blank_method == 'median'


# ---------------------------------------------------------------------------
# fit_treatment and results_table
# ---------------------------------------------------------------------------

def test_fit_treatment_returns_list():
    volume_matrix = [
        VOLUME,
        [0, 7.9, 15.8, 25.1, 37.2, 59.8, 113.1, 141.0, 147.5, 149.0]
    ]
    results = fit_treatment(
        TIME,
        np.array(volume_matrix).T,
        treatment_name="Test",
        verbose=False
    )
    assert len(results) == 2
    assert all(r.r_squared > 0.95 for r in results)

def test_results_table_shape():
    pytest.importorskip("pandas")
    volume_matrix = [
        VOLUME,
        [0, 7.9, 15.8, 25.1, 37.2, 59.8, 113.1, 141.0, 147.5, 149.0]
    ]
    results = fit_treatment(
        TIME,
        np.array(volume_matrix).T,
        treatment_name="Test",
        verbose=False
    )
    df = results_table(results)
    assert len(df) == 2
    assert "R²" in df.columns
    assert "Vf (mL)" in df.columns
