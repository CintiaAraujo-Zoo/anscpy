"""
anscpy.gas._results
====================
Result class for in vitro gas production model fitting.

The GasProductionResult object is returned by fit_gas_production()
and contains estimated parameters, goodness-of-fit metrics,
and methods for summary and visualization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GasProductionResult:
    """Complete result from fitting an in vitro gas production model.

    Attributes
    ----------
    model_name : str
        Name of the fitted model (e.g., 'LE0').
    param_names : list
        Names of the estimated parameters.
    popt : np.ndarray
        Estimated parameter values.
    se : np.ndarray
        Standard errors of the estimated parameters.
    r_squared : float
        Coefficient of determination (R²).
    rmse : float
        Root mean squared error (mL).
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    n_obs : int
        Number of observations used in fitting.
    time : np.ndarray
        Incubation time vector (h).
    observed : np.ndarray
        Blank-corrected volumes used in fitting (mL).
    observed_raw : np.ndarray
        Original volumes before blank correction (mL).
    predicted : np.ndarray
        Model-predicted volumes at each timepoint (mL).
    func : callable
        The fitted model function.
    blank_used : np.ndarray or None
        Aggregated blank series used for correction (mL).
    blank_method : str
        Aggregation method used for blank correction.
    treatment : str
        Treatment name for identification.
    replicate : str
        Replicate identifier.
    converged : bool
        Whether the fitting converged successfully.
    """

    model_name:   str
    param_names:  list
    popt:         np.ndarray
    se:           np.ndarray
    r_squared:    float
    rmse:         float
    aic:          float
    bic:          float
    n_obs:        int
    time:         np.ndarray
    observed:     np.ndarray
    observed_raw: np.ndarray
    predicted:    np.ndarray
    func:         object
    blank_used:   object = None
    blank_method: str = ""
    treatment:    str = ""
    replicate:    str = ""
    converged:    bool = True

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def Vf(self) -> float:
        """Asymptotic total gas volume (mL)."""
        return float(self.popt[0])

    @property
    def r2(self) -> float:
        """Coefficient of determination (R²)."""
        return self.r_squared

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a formatted summary of the fitting results."""
        sep  = "=" * 66
        sep2 = "-" * 54
        print(sep)
        print(f"  MODEL: {self.model_name}  —  In Vitro Gas Production Kinetics")
        print(sep)

        if self.treatment or self.replicate:
            if self.treatment:
                print(f"  Treatment  : {self.treatment}")
            if self.replicate:
                print(f"  Replicate  : {self.replicate}")
            print()

        if self.blank_used is not None:
            print(f"  Blank correction : YES  (method: {self.blank_method})")
            print()

        print("  Estimated parameters:")
        print("  " + sep2)
        print(f"  {'Parameter':<26} {'Estimate':>12} {'SE':>10}")
        print("  " + sep2)
        for i, name in enumerate(self.param_names):
            print(f"  {name:<26} {self.popt[i]:>12.4f} {self.se[i]:>10.4f}")
        print("  " + sep2)
        print()
        print("  Goodness of fit:")
        print("  " + sep2)
        print(f"  R²   = {self.r_squared:.6f}")
        print(f"  RMSE = {self.rmse:.4f} mL")
        print(f"  AIC  = {self.aic:.2f}")
        print(f"  BIC  = {self.bic:.2f}")
        print(f"  n    = {self.n_obs} observations")
        print(sep)

    # ── Plot ──────────────────────────────────────────────────────────────

    def plot(self,
             title: Optional[str] = None,
             save_path: Optional[str] = None,
             dpi: int = 300,
             show: bool = True,
             show_blank: bool = True):
        """
        Generate a publication-quality fitting plot.

        Parameters
        ----------
        title : str, optional
            Plot title. Auto-generated if not provided.
        save_path : str, optional
            File path to save the figure (e.g., 'fit_result.png').
        dpi : int, default 300
            Resolution of the saved figure.
        show : bool, default True
            If True, display the plot interactively.
        show_blank : bool, default True
            If True and blank correction was applied, adds a lower panel
            showing the aggregated blank series used.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        t_smooth = np.linspace(0, self.time.max() * 1.05, 600)
        v_fit    = self.func(t_smooth, *self.popt)

        show_blank_panel = (
            show_blank
            and self.blank_used is not None
            and len(self.blank_used) == len(self.time)
        )

        if show_blank_panel:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 7),
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
        else:
            fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.scatter(
            self.time, self.observed,
            c="#2C3E50", s=55, zorder=5,
            label="Observed (corrected)",
            edgecolors="white", linewidths=0.6
        )
        ax1.plot(
            t_smooth, v_fit,
            color="#E74C3C", lw=2.5,
            label=f"{self.model_name}  R² = {self.r_squared:.4f}"
        )

        if (self.blank_used is not None
                and not np.array_equal(self.observed, self.observed_raw)):
            ax1.scatter(
                self.time, self.observed_raw,
                c="#BDC3C7", s=30, zorder=3, alpha=0.6,
                label="Observed (raw)", marker='x', linewidths=1.2
            )

        _title = title or f"{self.model_name} Model Fit"
        if self.treatment:
            _title += f"\n{self.treatment}"
            if self.replicate:
                _title += f" — {self.replicate}"

        ax1.set_title(_title, fontsize=12, fontweight="bold")
        ax1.set_ylabel("Cumulative gas volume (mL)", fontsize=11)
        ax1.legend(framealpha=0.9, fontsize=9, loc="lower right")
        ax1.set_xlim(0, None)
        ax1.set_ylim(0, None)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        if show_blank_panel:
            ax2.bar(
                self.time, self.blank_used,
                color="#3498DB", alpha=0.6, width=1.5,
                label="Aggregated blank"
            )
            ax2.set_ylabel("Blank (mL)", fontsize=9)
            ax2.set_xlabel("Incubation time (h)", fontsize=11)
            ax2.legend(fontsize=8, loc="upper left")
            ax2.grid(True, alpha=0.2)
            ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        else:
            ax1.set_xlabel("Incubation time (h)", fontsize=11)
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  Figure saved to: {save_path}")
        if show:
            plt.show()

        return fig
