"""
Apex Analytics — Calibration Check
Computes Brier score, log loss, and calibration curve from backtest results.

A well-calibrated model: when it says 60%, the team wins ~60% of the time.
A perfectly calibrated model has all points on the diagonal of the calibration plot.

Calibration quality thresholds (per the build plan):
  - Max deviation per bucket: ≤ 5% (0.05)
  - Brier score: < 0.240 (beat FiveThirtyEight's ~0.243)
  - Accuracy: > 55% on held-out season

Usage:
  python -m backtesting.calibration_check                       # Check 2024 backtest
  python -m backtesting.calibration_check --year 2024          # Explicit year
  python -m backtesting.calibration_check --fit-calibrator     # Fit isotonic calibrator from results
  python -m backtesting.calibration_check --plot               # Save calibration curve PNG
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_brier_score(probs: List[float], actuals: List[int]) -> float:
    """
    Brier Score = mean((prob - actual)^2).
    Range: 0 (perfect) to 1 (worst).
    MLB baseline (always predict 50%): 0.250.
    Well-calibrated target: < 0.220 long-term.
    """
    if not probs:
        return 0.0
    return sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs)


def compute_log_loss(probs: List[float], actuals: List[int], eps: float = 1e-9) -> float:
    """
    Binary cross-entropy loss.
    Penalizes overconfident wrong predictions more than Brier score.
    """
    if not probs:
        return 0.0
    total = 0.0
    for p, a in zip(probs, actuals):
        p_clipped = max(eps, min(1.0 - eps, p))
        total += -(a * math.log(p_clipped) + (1 - a) * math.log(1 - p_clipped))
    return total / len(probs)


def compute_accuracy(probs: List[float], actuals: List[int], threshold: float = 0.50) -> float:
    """Fraction of games where prediction direction (≥ threshold) was correct."""
    if not probs:
        return 0.0
    correct = sum(
        int((p >= threshold) == bool(a))
        for p, a in zip(probs, actuals)
    )
    return correct / len(probs)


def compute_resolution(probs: List[float], actuals: List[int]) -> float:
    """
    Resolution = how much predictions spread from mean.
    Higher = more decisive predictions (good if calibrated).
    Brier = Uncertainty - Resolution + Reliability.
    """
    if not probs:
        return 0.0
    mean_base = sum(actuals) / len(actuals)
    return sum((p - mean_base) ** 2 for p in probs) / len(probs)


def calibration_curve(
    probs:   List[float],
    actuals: List[int],
    n_bins:  int = 10,
) -> Dict:
    """
    Build calibration curve data by bucketing predictions into equal-width bins.

    Returns:
      {
        bins:          list of (low, high, center) tuples
        predicted:     mean predicted probability per bin
        actual:        actual win rate per bin
        counts:        number of games per bin
        max_error:     max |predicted - actual| across bins
        mean_error:    mean |predicted - actual| across bins
        passes:        bool — True if max_error ≤ 0.05
      }
    """
    import numpy as np

    bins_data    = []
    bin_edges    = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        in_bin  = [(p, a) for p, a in zip(probs, actuals) if lo <= p < hi]

        if not in_bin:
            # Include empty bin for display purposes
            bins_data.append({
                "low":       round(lo, 3),
                "high":      round(hi, 3),
                "center":    round((lo + hi) / 2, 3),
                "predicted": round((lo + hi) / 2, 3),
                "actual":    None,
                "count":     0,
            })
            continue

        bin_probs   = [x[0] for x in in_bin]
        bin_acts    = [x[1] for x in in_bin]
        mean_pred   = sum(bin_probs) / len(bin_probs)
        mean_actual = sum(bin_acts)  / len(bin_acts)

        bins_data.append({
            "low":       round(lo, 3),
            "high":      round(hi, 3),
            "center":    round((lo + hi) / 2, 3),
            "predicted": round(mean_pred, 4),
            "actual":    round(mean_actual, 4),
            "count":     len(in_bin),
        })

    # Compute errors only on populated bins
    populated = [b for b in bins_data if b["actual"] is not None]
    if populated:
        errors     = [abs(b["predicted"] - b["actual"]) for b in populated]
        max_err    = max(errors)
        mean_err   = sum(errors) / len(errors)
    else:
        max_err  = 0.0
        mean_err = 0.0

    return {
        "bins":       bins_data,
        "max_error":  round(max_err, 4),
        "mean_error": round(mean_err, 4),
        "n_bins":     n_bins,
        "n_populated": len(populated),
        "passes":     max_err <= 0.05,
        "target":     0.05,
    }


def reliability_diagram_text(curve: Dict) -> str:
    """
    ASCII reliability diagram.
    Perfect calibration = dots on the diagonal.
    """
    lines = ["Reliability Diagram (calibration curve)"]
    lines.append("  Pred   Actual  Count   Error   Bar")
    lines.append("  " + "─" * 50)

    for b in curve["bins"]:
        if b["actual"] is None:
            lines.append(f"  {b['center']:.2f}   {'(no data)':>8}")
            continue

        err   = abs(b["predicted"] - b["actual"])
        flag  = "⚠" if err > 0.05 else " "
        bar_p = "█" * int(b["predicted"] * 20)   # predicted
        bar_a = "░" * int(b["actual"]    * 20)   # actual overlay

        lines.append(
            f"  {b['predicted']:.3f}  {b['actual']:.3f}   "
            f"{b['count']:>5}   {err:.3f}{flag}   "
            f"|{bar_p:<20}| (pred)"
        )

    lines.append(f"\n  Max deviation: {curve['max_error']:.4f}  "
                 f"(target ≤ 0.05) — {'PASS ✓' if curve['passes'] else 'FAIL — needs calibration'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load backtest results from CSV
# ---------------------------------------------------------------------------

def load_backtest_results(year: int) -> Tuple[List[float], List[int], List[dict]]:
    """
    Load backtest results CSV.
    Returns (calibrated_probs, actuals, raw_rows).
    """
    from config import HISTORICAL_DIR
    path = HISTORICAL_DIR / f"backtest_results_{year}.csv"

    if not path.exists():
        logger.error(
            "Backtest results not found: %s\n"
            "Run: python -m backtesting.backtest_runner --year %d",
            path, year,
        )
        return [], [], []

    rows  = []
    probs = []
    acts  = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p = float(row["calibrated_prob"])
                a = int(row["home_win"])
                probs.append(p)
                acts.append(a)
                rows.append(row)
            except (KeyError, ValueError):
                continue

    logger.info("Loaded %d backtest results for %d", len(rows), year)
    return probs, acts, rows


# ---------------------------------------------------------------------------
# Full calibration report
# ---------------------------------------------------------------------------

def run_calibration_report(
    results:  Optional[List[dict]] = None,
    year:     int                  = 2024,
    n_bins:   int                  = 10,
) -> Dict:
    """
    Compute and log the full calibration report.

    Parameters
    ----------
    results : Optional list of result dicts from backtest_runner.run_backtest().
              If None, loads from the CSV file for `year`.
    year    : Season year (used for CSV loading if results is None).

    Returns dict with all metrics.
    """
    if results is not None:
        probs   = [r["calibrated_prob"] for r in results]
        actuals = [r["home_win"]        for r in results]
    else:
        probs, actuals, _ = load_backtest_results(year)

    if not probs:
        logger.warning("No results available for calibration report.")
        return {}

    n = len(probs)

    # Core metrics
    brier    = compute_brier_score(probs, actuals)
    ll       = compute_log_loss(probs, actuals)
    acc      = compute_accuracy(probs, actuals)
    home_rate = sum(actuals) / n
    res      = compute_resolution(probs, actuals)

    # Calibration curve
    curve    = calibration_curve(probs, actuals, n_bins=n_bins)

    # Per-layer accuracy (if results provided)
    layer_metrics = {}
    if results:
        for layer in ["mc_prob", "elo_prob", "rf_prob", "lr_prob", "raw_prob"]:
            layer_probs = [r.get(layer, 0.5) for r in results]
            try:
                layer_probs = [float(p) for p in layer_probs]
                layer_metrics[layer] = {
                    "accuracy":    round(compute_accuracy(layer_probs, actuals), 4),
                    "brier_score": round(compute_brier_score(layer_probs, actuals), 4),
                }
            except Exception:
                pass

    report = {
        "year":                year,
        "n_games":             n,
        "home_win_rate":       round(home_rate, 4),
        "brier_score":         round(brier, 4),
        "log_loss":            round(ll, 4),
        "accuracy":            round(acc, 4),
        "resolution":          round(res, 4),
        "calibration_curve":   curve,
        "layer_comparison":    layer_metrics,
        "targets": {
            "brier_target":    0.240,
            "accuracy_target": 0.550,
            "cal_max_error":   0.050,
        },
        "pass_fail": {
            "brier":       brier    < 0.240,
            "accuracy":    acc      > 0.550,
            "calibration": curve["passes"],
        },
    }

    # Print report
    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATION REPORT — %d Season", year)
    logger.info("=" * 60)
    logger.info("  Games analyzed:   %d", n)
    logger.info("  Home win rate:    %.1f%%", home_rate * 100)
    logger.info("")
    logger.info("  ACCURACY METRICS")
    logger.info("  ─────────────────────────────────────────")
    logger.info("  Accuracy:    %.1f%%  target >55%%  — %s",
                acc * 100, "PASS ✓" if acc > 0.55 else "MISS ✗")
    logger.info("  Brier Score: %.4f  target <0.240 — %s",
                brier, "PASS ✓" if brier < 0.240 else "MISS ✗")
    logger.info("  Log Loss:    %.4f", ll)
    logger.info("  AUC-ROC:     (see layer comparison below)")
    logger.info("  Resolution:  %.4f  (higher = more decisive)", res)
    logger.info("")

    if layer_metrics:
        logger.info("  LAYER COMPARISON")
        logger.info("  ─────────────────────────────────────────")
        logger.info("  %-15s  Accuracy   Brier", "Layer")
        for layer, m in layer_metrics.items():
            label = layer.replace("_prob", "").upper()
            logger.info(
                "  %-15s  %.1f%%      %.4f",
                label, m["accuracy"] * 100, m["brier_score"],
            )
        logger.info("")

    logger.info(reliability_diagram_text(curve))

    # Save to JSON
    _save_calibration_report(report, year)

    return report


def _save_calibration_report(report: dict, year: int) -> None:
    """Save calibration report JSON to data/historical/."""
    from config import HISTORICAL_DIR
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HISTORICAL_DIR / f"calibration_report_{year}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Calibration report saved → %s", out_path)


# ---------------------------------------------------------------------------
# Fit calibrator from backtest results
# ---------------------------------------------------------------------------

def fit_calibrator_from_backtest(year: int = 2024) -> Optional[Dict]:
    """
    Load backtest results and use them to fit the probability calibrator.

    This uses the raw (pre-calibration) ensemble outputs as inputs and
    the actual game outcomes as targets. Saves the fitted calibrator.
    """
    from config import HISTORICAL_DIR
    path = HISTORICAL_DIR / f"backtest_results_{year}.csv"

    if not path.exists():
        logger.error("Backtest results not found — run backtest first.")
        return None

    raw_probs = []
    actuals   = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                raw_probs.append(float(row["raw_prob"]))
                actuals.append(int(row["home_win"]))
            except (KeyError, ValueError):
                continue

    if len(raw_probs) < 50:
        logger.warning("Too few backtest results (%d) to fit calibrator.", len(raw_probs))
        return None

    logger.info("Fitting calibrator on %d backtest games (raw → actual)...", len(raw_probs))

    try:
        import numpy as np
        from ensemble.calibrator import get_calibrator
        cal = get_calibrator()
        metrics = cal.fit(np.array(raw_probs), np.array(actuals))
        cal.save()
        logger.info("Calibrator fitted and saved. Method: %s", metrics.get("method"))
        return metrics
    except Exception as exc:
        logger.error("Failed to fit calibrator: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Calibration curve plot (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_calibration_curve(year: int = 2024, output_path: Optional[Path] = None) -> Optional[str]:
    """
    Generate a calibration curve PNG using matplotlib.
    Returns path to saved PNG or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed — skipping calibration plot.")
        return None

    probs, actuals, _ = load_backtest_results(year)
    if not probs:
        return None

    curve = calibration_curve(probs, actuals, n_bins=10)
    populated = [b for b in curve["bins"] if b["actual"] is not None]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], color="#444466", linestyle="--", linewidth=1.0, label="Perfect calibration")

    # Model calibration curve
    pred_centers = [b["predicted"] for b in populated]
    act_rates    = [b["actual"]    for b in populated]
    counts       = [b["count"]     for b in populated]

    # Scatter with size proportional to game count
    sizes = [max(30, min(300, c * 0.8)) for c in counts]
    ax.scatter(pred_centers, act_rates, s=sizes, color="#4a9eff", zorder=3, label="Model (calibrated)")
    ax.plot(pred_centers, act_rates, color="#4a9eff", alpha=0.6, linewidth=1.5)

    # Error bands
    for pred, actual, cnt in zip(pred_centers, act_rates, counts):
        err = abs(pred - actual)
        color = "#ff6b6b" if err > 0.05 else "#4caf76"
        ax.vlines(pred, min(pred, actual), max(pred, actual),
                  color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel("Predicted probability (home win)", color="#cccccc", fontsize=11)
    ax.set_ylabel("Actual win rate",                  color="#cccccc", fontsize=11)
    ax.set_title(
        f"Calibration Curve — {year} Season\n"
        f"Brier={compute_brier_score(probs, actuals):.4f}  "
        f"Acc={compute_accuracy(probs, actuals):.1%}  "
        f"MaxErr={curve['max_error']:.3f}",
        color="#ffffff", fontsize=12, pad=12,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#444466")

    good_patch = mpatches.Patch(color="#4caf76", label="Error ≤ 5%")
    warn_patch = mpatches.Patch(color="#ff6b6b", label="Error > 5%")
    ax.legend(handles=[good_patch, warn_patch], facecolor="#1a1a2e", labelcolor="#cccccc")

    if output_path is None:
        from config import HISTORICAL_DIR
        HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
        output_path = HISTORICAL_DIR / f"calibration_curve_{year}.png"

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Calibration curve saved → %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Apex Analytics calibration check")
    parser.add_argument("--year",           type=int, default=2024)
    parser.add_argument("--fit-calibrator", action="store_true",
                        help="Fit and save the ensemble calibrator from backtest results")
    parser.add_argument("--plot",           action="store_true",
                        help="Save calibration curve PNG")
    parser.add_argument("--n-bins",         type=int, default=10)
    args = parser.parse_args()

    run_calibration_report(year=args.year, n_bins=args.n_bins)

    if args.fit_calibrator:
        metrics = fit_calibrator_from_backtest(args.year)
        if metrics:
            logger.info("Calibrator fitted: method=%s  n=%d  Brier=%.4f",
                        metrics.get("method"), metrics.get("n_games", 0),
                        metrics.get("brier_score", 0))

    if args.plot:
        plot_calibration_curve(args.year)


if __name__ == "__main__":
    main()
