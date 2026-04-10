"""
Apex Analytics — Backtest Runner
Runs the full prediction pipeline over a historical date range and compares
predicted win probabilities against actual game outcomes.

This produces a calibration dataset that is used to:
  1. Evaluate model accuracy (Brier score, log loss, accuracy)
  2. Fit the isotonic regression calibrator (once 1,000+ games accumulate)
  3. Track model performance over time

The backtest does NOT use the live data pipeline (which requires real-time API calls).
Instead it uses:
  - Historical game outcomes from the prebuilt CSV datasets
  - The trained RF and LR models
  - Elo ratings simulated game-by-game
  - The ensemble blender for combining predictions

Usage:
  python -m backtesting.backtest_runner                          # Full 2024 season
  python -m backtesting.backtest_runner --year 2024             # Explicit year
  python -m backtesting.backtest_runner --start 2024-04-01 --end 2024-06-30
  python -m backtesting.backtest_runner --store-db              # Store results in CalibrationHistory
  python -m backtesting.backtest_runner --report                # Print metrics after run

Output:
  data/historical/backtest_results_YYYY.csv
"""

import argparse
import csv
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature name constants (must match ensemble layer exactly)
# ---------------------------------------------------------------------------

RF_FEATURES = [
    "home_starter_siera", "away_starter_siera",
    "home_starter_decay_xera", "away_starter_decay_xera",
    "home_team_xwoba", "away_team_xwoba",
    "home_bullpen_xfip", "away_bullpen_xfip",
    "park_factor_runs", "park_factor_hr",
    "home_decay_win_pct", "away_decay_win_pct",
    "weather_run_adj",
    "home_elo", "away_elo",
]

LR_FEATURES = [
    "home_starter_era", "away_starter_era",
    "home_team_xwoba", "away_team_xwoba",
    "home_bullpen_xfip", "away_bullpen_xfip",
    "park_factor_runs",
    "home_decay_win_pct", "away_decay_win_pct",
    "weather_run_adj",
    "home_starter_decay_xera", "away_starter_decay_xera",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_backtest_data(year: int) -> Optional[List[dict]]:
    """
    Load the historical dataset CSV for the given year.
    Returns list of row dicts — one per game.
    """
    from config import HISTORICAL_DIR

    path = HISTORICAL_DIR / f"training_data_{year}.csv"
    if not path.exists():
        logger.error(
            "Backtest data not found: %s\n"
            "Run: python -m scripts.build_historical_dataset --year %d",
            path, year,
        )
        return None

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric columns
            for col in [
                "home_starter_siera", "away_starter_siera",
                "home_starter_decay_xera", "away_starter_decay_xera",
                "home_starter_era", "away_starter_era",
                "home_team_xwoba", "away_team_xwoba",
                "home_bullpen_xfip", "away_bullpen_xfip",
                "park_factor_runs", "park_factor_hr",
                "home_decay_win_pct", "away_decay_win_pct",
                "weather_run_adj", "home_elo", "away_elo",
            ]:
                try:
                    row[col] = float(row[col])
                except (TypeError, ValueError):
                    row[col] = 0.0

            for col in ["home_win", "home_score", "away_score", "home_id", "away_id"]:
                try:
                    row[col] = int(float(row[col]))
                except (TypeError, ValueError):
                    row[col] = 0

            rows.append(row)

    logger.info("Loaded %d games from %s", len(rows), path.name)
    return rows


def filter_by_date_range(
    rows: List[dict],
    start: Optional[str],
    end:   Optional[str],
) -> List[dict]:
    """Filter rows to a date range (YYYY-MM-DD strings, inclusive)."""
    if not start and not end:
        return rows

    filtered = []
    for r in rows:
        gd = r.get("game_date", "")
        if start and gd < start:
            continue
        if end and gd > end:
            continue
        filtered.append(r)

    logger.info(
        "Date filter [%s → %s]: %d games",
        start or "season start", end or "season end", len(filtered),
    )
    return filtered


# ---------------------------------------------------------------------------
# Ensemble prediction on a single game row
# ---------------------------------------------------------------------------

def _predict_single_game(row: dict, rf_model, lr_model) -> dict:
    """
    Run ensemble prediction on a single historical game row.
    Returns dict with all layer outputs + ensemble probability.
    """
    from ensemble.elo_system import win_probability as elo_win_probability
    from ensemble.blender import get_phase_weights_for_date
    from config import ELO_HOME_FIELD_BONUS

    game_date_str = row.get("game_date", "")
    try:
        game_date = date.fromisoformat(game_date_str)
    except ValueError:
        game_date = date.today()

    # ── Elo layer ───────────────────────────────────────────
    home_elo = float(row.get("home_elo", 1500))
    away_elo = float(row.get("away_elo", 1500))
    elo_prob = elo_win_probability(home_elo, away_elo)

    # ── Random Forest layer ─────────────────────────────────
    rf_features = {k: float(row.get(k, 0.0)) for k in RF_FEATURES}
    try:
        rf_prob = rf_model.predict_win_probability(rf_features)
    except Exception:
        rf_prob = 0.53  # Fallback

    # ── Logistic Regression layer ───────────────────────────
    lr_features = {k: float(row.get(k, 0.0)) for k in LR_FEATURES}
    try:
        lr_prob = lr_model.predict_win_probability(lr_features)
    except Exception:
        lr_prob = 0.53  # Fallback

    # ── Monte Carlo placeholder (use Elo + ERA proxy) ───────
    # True Monte Carlo can't run retroactively without full player profiles.
    # Use a weighted Elo + ERA differential proxy as Monte Carlo stand-in.
    home_era = float(row.get("home_starter_era", 4.20))
    away_era = float(row.get("away_starter_era", 4.20))
    era_diff  = (away_era - home_era) / 4.20   # Positive = home advantage
    mc_prob   = float(np.clip(elo_prob + era_diff * 0.04, 0.35, 0.70))

    # ── Season-phase weights ────────────────────────────────
    weights = get_phase_weights_for_date(game_date)
    w_mc  = weights.get("mc",  0.50)
    w_elo = weights.get("elo", 0.15)
    w_rf  = weights.get("rf",  0.25)
    w_lr  = weights.get("lr",  0.10)

    raw_prob = (
        mc_prob  * w_mc +
        elo_prob * w_elo +
        rf_prob  * w_rf +
        lr_prob  * w_lr
    )
    raw_prob = float(np.clip(raw_prob, 0.05, 0.95))

    # ── Calibration ─────────────────────────────────────────
    try:
        from ensemble.calibrator import calibrate
        calibrated_prob = calibrate(raw_prob)
    except Exception:
        calibrated_prob = raw_prob

    return {
        "game_pk":        row.get("game_pk"),
        "game_date":      game_date_str,
        "home_abbr":      row.get("home_abbr", ""),
        "away_abbr":      row.get("away_abbr", ""),
        "mc_prob":        round(mc_prob, 4),
        "elo_prob":       round(elo_prob, 4),
        "rf_prob":        round(rf_prob, 4),
        "lr_prob":        round(lr_prob, 4),
        "raw_prob":       round(raw_prob, 4),
        "calibrated_prob": round(calibrated_prob, 4),
        "home_win":       int(row.get("home_win", 0)),
        "correct_raw":    int((raw_prob >= 0.50) == bool(row.get("home_win", 0))),
        "correct_cal":    int((calibrated_prob >= 0.50) == bool(row.get("home_win", 0))),
    }


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    year:     int                = 2024,
    start:    Optional[str]      = None,
    end:      Optional[str]      = None,
    store_db: bool               = False,
) -> List[dict]:
    """
    Run full backtest for a given year (and optional date range).

    Returns list of result dicts — one per game.
    """
    from config import HISTORICAL_DIR

    rows = load_backtest_data(year)
    if not rows:
        return []

    rows = filter_by_date_range(rows, start, end)
    if not rows:
        logger.warning("No games in selected date range.")
        return []

    # Load models
    logger.info("Loading ensemble models...")
    try:
        from ensemble.random_forest_model import get_rf_model
        rf_model = get_rf_model()
        if not rf_model._is_trained:
            logger.warning("RF model not trained — using baseline predictions.")
    except Exception as exc:
        logger.warning("Could not load RF model: %s — using baseline", exc)
        rf_model = _DummyModel(0.53)

    try:
        from ensemble.logistic_model import get_lr_model
        lr_model = get_lr_model()
        if not lr_model._is_trained:
            logger.warning("LR model not trained — using baseline predictions.")
    except Exception as exc:
        logger.warning("Could not load LR model: %s — using baseline", exc)
        lr_model = _DummyModel(0.53)

    # Run predictions
    results = []
    logger.info("Running predictions for %d games...", len(rows))

    for i, row in enumerate(rows):
        try:
            result = _predict_single_game(row, rf_model, lr_model)
            results.append(result)
        except Exception as exc:
            logger.debug("Skipped game %s: %s", row.get("game_pk"), exc)

        if (i + 1) % 500 == 0:
            n_done = i + 1
            acc_so_far = sum(r["correct_cal"] for r in results) / len(results)
            logger.info(
                "Progress: %d / %d games  |  Running accuracy: %.1f%%",
                n_done, len(rows), acc_so_far * 100,
            )

    # Save results CSV
    if results:
        _save_results(results, year, HISTORICAL_DIR)

        if store_db:
            _store_in_db(results)

    return results


def _save_results(results: List[dict], year: int, output_dir: Path) -> None:
    """Save backtest results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"backtest_results_{year}.csv"

    fieldnames = [
        "game_pk", "game_date", "home_abbr", "away_abbr",
        "mc_prob", "elo_prob", "rf_prob", "lr_prob",
        "raw_prob", "calibrated_prob",
        "home_win", "correct_raw", "correct_cal",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Backtest results saved → %s", out_path)


def _store_in_db(results: List[dict]) -> None:
    """Store predictions in CalibrationHistory table for future calibrator fitting."""
    try:
        from data.cache.db import CalibrationHistory, get_session

        with get_session() as session:
            stored = 0
            for r in results:
                # Check if already exists
                existing = session.query(CalibrationHistory).filter(
                    CalibrationHistory.game_pk == r["game_pk"]
                ).first()

                if existing:
                    existing.actual_outcome = r["home_win"]
                    continue

                row = CalibrationHistory(
                    game_pk          = r["game_pk"],
                    game_date        = date.fromisoformat(r["game_date"]),
                    home_win_prob    = r["calibrated_prob"],
                    actual_outcome   = r["home_win"],
                    model_version    = "backtest_v1.0",
                    ensemble_raw     = r["raw_prob"],
                )
                session.add(row)
                stored += 1

        logger.info("Stored %d backtest results in CalibrationHistory DB.", stored)

    except Exception as exc:
        logger.warning("Could not store results in DB: %s", exc)


# ---------------------------------------------------------------------------
# Dummy model for when real models aren't trained
# ---------------------------------------------------------------------------

class _DummyModel:
    """Returns a constant probability — for graceful degradation."""
    _is_trained = False

    def __init__(self, p: float = 0.53):
        self._p = p

    def predict_win_probability(self, features: dict) -> float:
        return self._p


# ---------------------------------------------------------------------------
# Quick summary print
# ---------------------------------------------------------------------------

def print_backtest_summary(results: List[dict]) -> None:
    """Print a quick accuracy summary from backtest results."""
    if not results:
        logger.warning("No results to summarize.")
        return

    n           = len(results)
    n_correct   = sum(r["correct_cal"] for r in results)
    acc         = n_correct / n
    home_wins   = sum(r["home_win"] for r in results)
    home_rate   = home_wins / n

    probs   = [r["calibrated_prob"] for r in results]
    actuals = [r["home_win"] for r in results]
    brier   = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / n

    logger.info("")
    logger.info("=" * 55)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 55)
    logger.info("  Total games:     %d", n)
    logger.info("  Correct (cal):   %d (%.1f%%)", n_correct, acc * 100)
    logger.info("  Home win rate:   %.1f%%", home_rate * 100)
    logger.info("  Brier Score:     %.4f  (target: < 0.240)", brier)
    logger.info("  Vegas benchmark: ~58.2%% accuracy")
    logger.info("")

    # Per-month breakdown
    from collections import defaultdict
    monthly: Dict[str, List] = defaultdict(list)
    for r in results:
        month = r["game_date"][:7]
        monthly[month].append((r["correct_cal"], r["calibrated_prob"], r["home_win"]))

    logger.info("  Monthly accuracy:")
    for month in sorted(monthly):
        games = monthly[month]
        m_acc   = sum(g[0] for g in games) / len(games)
        m_brier = sum((g[1] - g[2]) ** 2 for g in games) / len(games)
        logger.info(
            "    %s:  %.1f%%  Brier=%.4f  n=%d",
            month, m_acc * 100, m_brier, len(games),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Apex Analytics backtest")
    parser.add_argument("--year",     type=int, default=2024, help="Season to backtest (default: 2024)")
    parser.add_argument("--start",    type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--store-db", action="store_true", help="Store results in CalibrationHistory DB")
    parser.add_argument("--report",   action="store_true", help="Print detailed accuracy report after run")
    args = parser.parse_args()

    results = run_backtest(
        year     = args.year,
        start    = args.start,
        end      = args.end,
        store_db = args.store_db,
    )

    if results:
        print_backtest_summary(results)

        if args.report:
            from backtesting.calibration_check import run_calibration_report
            report = run_calibration_report(results)
            logger.info("Detailed calibration report saved.")


if __name__ == "__main__":
    main()
