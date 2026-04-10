"""
Apex Analytics — Chart Generator
Produces base64-encoded PNG charts for embedding directly in HTML reports.
No file I/O — all output is base64 strings for <img src="data:image/png;base64,...">

Charts:
  - run_distribution_chart(): histogram of total runs from Monte Carlo percentiles
  - win_probability_chart(): horizontal probability bars for home vs away
  - team_run_distribution_chart(): separate home/away run distribution overlay

Design: dark sports-analytics aesthetic
  Background: #0d0d1a (deep navy)
  Home accent: #4a9eff (blue)
  Away accent: #ff6b6b (coral red)
  Grid: #2a2a3e (subtle)
  Text: #e0e0e0 (light grey)
"""

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Color palette
BG_COLOR      = "#0d0d1a"
PANEL_COLOR   = "#1a1a2e"
HOME_COLOR    = "#4a9eff"
AWAY_COLOR    = "#ff6b6b"
GRID_COLOR    = "#2a2a3e"
TEXT_COLOR    = "#e0e0e0"
ACCENT_COLOR  = "#ffd700"  # Gold for median/key lines

# Standard percentiles from Monte Carlo
PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


def run_distribution_chart(
    run_dist:     dict,
    home_abbr:    str = "HOME",
    away_abbr:    str = "AWAY",
    width_in:     float = 7.0,
    height_in:    float = 3.2,
) -> str:
    """
    Horizontal bar chart showing the run distribution from Monte Carlo simulation.
    Uses the percentile dict from monte_carlo.run_monte_carlo()["run_distribution"].

    Parameters
    ----------
    run_dist  : dict — {5: x, 10: x, 25: x, 50: x, 75: x, 90: x, 95: x}
    home_abbr : Home team abbreviation (for title)
    away_abbr : Away team abbreviation (for title)

    Returns
    -------
    str — base64-encoded PNG string.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, ax = plt.subplots(figsize=(width_in, height_in))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(PANEL_COLOR)

        # Build synthetic distribution from percentiles using linear interpolation
        pcts   = [5, 10, 25, 50, 75, 90, 95]
        values = [run_dist.get(p, 0) for p in pcts]

        # Create a smooth bar chart representing the distribution shape
        # X-axis: total runs (0 to max), Y-axis: relative density
        x_min = max(0, int(values[0]) - 1)
        x_max = int(values[-1]) + 2
        x     = np.arange(x_min, x_max + 1)

        # Use percentile-based density estimation
        density = _percentile_to_density(x, pcts, values)

        # Color bars: below median = away, above median = home (visual split)
        median = run_dist.get(50, 9)
        colors = [HOME_COLOR if xi >= median else AWAY_COLOR for xi in x]

        bars = ax.bar(x, density, color=colors, width=0.85, alpha=0.85, zorder=3)

        # Median line
        ax.axvline(
            median, color=ACCENT_COLOR, linewidth=1.8,
            linestyle="--", alpha=0.9, zorder=4, label=f"Median: {median:.0f}",
        )

        # Percentile markers on x-axis
        for pct, val in zip(pcts, values):
            if pct in (25, 75):
                ax.axvline(
                    val, color=GRID_COLOR, linewidth=1.0,
                    linestyle=":", alpha=0.7, zorder=2,
                )

        # Annotations
        ax.text(
            0.02, 0.95,
            f"P5: {values[0]:.0f}   P25: {values[2]:.0f}   Med: {values[3]:.0f}"
            f"   P75: {values[4]:.0f}   P95: {values[6]:.0f}",
            transform=ax.transAxes,
            color=TEXT_COLOR, fontsize=7.5, va="top", ha="left",
            fontfamily="monospace",
        )

        # Style
        ax.set_xlabel("Total Runs", color=TEXT_COLOR, fontsize=8)
        ax.set_ylabel("Likelihood", color=TEXT_COLOR, fontsize=8)
        ax.set_title(
            f"{away_abbr} @ {home_abbr} — Run Distribution (7,000 simulations)",
            color=TEXT_COLOR, fontsize=9, pad=6,
        )
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticklabels([])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.5)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)

        # Legend
        home_patch = mpatches.Patch(color=HOME_COLOR, label=f"≥ Median")
        away_patch = mpatches.Patch(color=AWAY_COLOR, label=f"< Median")
        med_line   = plt.Line2D([0], [0], color=ACCENT_COLOR, linewidth=1.8,
                                linestyle="--", label=f"Median {median:.0f}R")
        ax.legend(
            handles=[away_patch, med_line, home_patch],
            loc="upper right", fontsize=7,
            facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
        )

        fig.tight_layout(pad=0.4)
        return _fig_to_base64(fig)

    except Exception as exc:
        logger.warning("run_distribution_chart failed: %s", exc)
        return ""


def win_probability_chart(
    home_pct:  float,
    away_pct:  float,
    home_abbr: str = "HOME",
    away_abbr: str = "AWAY",
    ci_band:   Optional[tuple] = None,
    width_in:  float = 7.0,
    height_in: float = 1.8,
) -> str:
    """
    Horizontal stacked bar showing home vs away win probability.

    Parameters
    ----------
    home_pct  : Home win probability (0.0–1.0)
    away_pct  : Away win probability (0.0–1.0)
    ci_band   : Optional (lower, upper) confidence interval on home probability

    Returns
    -------
    str — base64-encoded PNG string.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, ax = plt.subplots(figsize=(width_in, height_in))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        bar_height = 0.55

        # Away bar (left portion)
        ax.barh(0, away_pct, height=bar_height, color=AWAY_COLOR, alpha=0.90, left=0)
        # Home bar (right portion)
        ax.barh(0, home_pct, height=bar_height, color=HOME_COLOR, alpha=0.90, left=away_pct)

        # Center divider
        ax.axvline(0.5, color="#555577", linewidth=1.2, linestyle="--", alpha=0.7)

        # Confidence interval band on home probability
        if ci_band and ci_band[0] < ci_band[1]:
            ci_lo, ci_hi = ci_band
            ax.barh(
                0, ci_hi - ci_lo, height=bar_height * 0.25,
                left=ci_lo, color=ACCENT_COLOR, alpha=0.7, zorder=5,
            )

        # Labels inside bars
        away_x = away_pct / 2
        home_x = away_pct + home_pct / 2

        ax.text(
            away_x, 0, f"{away_abbr}\n{away_pct*100:.1f}%",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold",
        )
        ax.text(
            home_x, 0, f"{home_abbr}\n{home_pct*100:.1f}%",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold",
        )

        # Edge labels
        ax.text(0.01, 0, away_abbr, ha="left", va="center",
                color=AWAY_COLOR, fontsize=7.5, transform=ax.transAxes)
        ax.text(0.99, 0, home_abbr, ha="right", va="center",
                color=HOME_COLOR, fontsize=7.5, transform=ax.transAxes)

        # Style
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        fig.tight_layout(pad=0.2)

        return _fig_to_base64(fig)

    except Exception as exc:
        logger.warning("win_probability_chart failed: %s", exc)
        return ""


def team_run_distribution_chart(
    home_run_dist: dict,
    away_run_dist: dict,
    home_abbr:     str = "HOME",
    away_abbr:     str = "AWAY",
    width_in:      float = 7.0,
    height_in:     float = 3.2,
) -> str:
    """
    Overlay histogram showing separate run distributions for home and away teams.

    Parameters
    ----------
    home_run_dist : dict — {5: x, ..., 95: x} for home runs scored
    away_run_dist : dict — {5: x, ..., 95: x} for away runs scored

    Returns
    -------
    str — base64-encoded PNG string.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(width_in, height_in))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(PANEL_COLOR)

        pcts = [5, 10, 25, 50, 75, 90, 95]

        home_vals = [home_run_dist.get(p, 0) for p in pcts]
        away_vals = [away_run_dist.get(p, 0) for p in pcts]

        x_min = 0
        x_max = max(int(home_vals[-1]), int(away_vals[-1])) + 2
        x     = np.arange(x_min, x_max + 1)

        home_density = _percentile_to_density(x, pcts, home_vals)
        away_density = _percentile_to_density(x, pcts, away_vals)

        ax.bar(x, home_density, width=0.85, color=HOME_COLOR, alpha=0.60,
               label=f"{home_abbr} runs", zorder=3)
        ax.bar(x, away_density, width=0.85, color=AWAY_COLOR, alpha=0.60,
               label=f"{away_abbr} runs", zorder=3)

        # Medians
        home_med = home_run_dist.get(50, 4.5)
        away_med = away_run_dist.get(50, 4.2)
        ax.axvline(home_med, color=HOME_COLOR, linewidth=1.8, linestyle="--",
                   alpha=0.95, label=f"{home_abbr} med: {home_med:.1f}")
        ax.axvline(away_med, color=AWAY_COLOR, linewidth=1.8, linestyle="--",
                   alpha=0.95, label=f"{away_abbr} med: {away_med:.1f}")

        ax.set_xlabel("Runs Scored", color=TEXT_COLOR, fontsize=8)
        ax.set_title(
            f"Team Run Distribution — {away_abbr} @ {home_abbr}",
            color=TEXT_COLOR, fontsize=9, pad=6,
        )
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticklabels([])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.5)
        ax.legend(
            fontsize=7.5, facecolor=PANEL_COLOR,
            edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
        )

        fig.tight_layout(pad=0.4)
        return _fig_to_base64(fig)

    except Exception as exc:
        logger.warning("team_run_distribution_chart failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_to_density(x: "np.ndarray", pcts: list, values: list) -> "np.ndarray":
    """
    Convert percentile breakpoints to a density-like array over integer x values.
    Uses linear interpolation between percentile anchors.
    """
    import numpy as np

    # Build CDF from percentile points
    cdf_x = [values[0]] + list(values) + [values[-1] + 3]
    cdf_y = [0.0] + [p / 100.0 for p in pcts] + [1.0]

    # Compute density as finite differences of the CDF
    density = np.zeros(len(x), dtype=float)
    for i, xi in enumerate(x):
        # CDF value at xi and xi+1
        cdf_at_x   = float(np.interp(xi,       cdf_x, cdf_y))
        cdf_at_xp1 = float(np.interp(xi + 1.0, cdf_x, cdf_y))
        density[i] = max(0.0, cdf_at_xp1 - cdf_at_x)

    # Normalize so max bar = 1.0 (visual only, no absolute scale)
    peak = density.max()
    if peak > 0:
        density /= peak
    return density


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded
