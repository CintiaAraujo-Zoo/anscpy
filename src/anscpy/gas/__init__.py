"""
anscpy.gas — In Vitro Gas Production Kinetics
==============================================
Fits mathematical models to cumulative gas production profiles
for ruminant nutrition research.

Available models
----------------
LE0  : Logistic-Exponential without lag (Wang et al., 2011) — DEFAULT
LEL  : Logistic-Exponential with lag (Wang et al., 2011)
MM   : Michaelis-Menten (Groot et al., 1996)
MIT  : Mitscherlich (France et al., 2000)
EXPL : Exponential with lag (Orskov & McDonald, 1979)
GOM  : Gompertz (Schofield et al., 1994)
LOG  : Logistic (Schofield et al., 1994)

Quick start
-----------
>>> from anscpy.gas import fit_gas_production
>>> time   = [0, 2, 4, 6, 8, 12, 24, 48, 72, 96]
>>> volume = [0, 10.3, 19.9, 28.9, 38.9, 61.9, 112.6, 140.1, 145.3, 149.1]
>>> result = fit_gas_production(time, volume)
>>> result.summary()
>>> result.plot()
"""

from ._fitting import fit_gas_production, fit_treatment, results_table
from ._correction import correct_blank
from ._results import GasProductionResult

__all__ = [
    "fit_gas_production",
    "fit_treatment",
    "results_table",
    "correct_blank",
    "GasProductionResult",
]
