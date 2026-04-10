"""
Apex Analytics — Accuracy Tracker
Ongoing daily accuracy and Brier score logging.

Reads from:
  1. DB: CalibrationHistory table (live production predictions)
  2. CSV: backtest_results_{year}.csv (historical backtest results)

Outputs:
  - Rolling 30-day Brier score and accuracy (used in report footer)
  - Season-to-date summary
  - Per-month breakdown
  - Trend detection (improving / degrading / stable)

This module is called:
  - In report/generator.py to populate the "DAILY MODEL PERFORMANCE" section
  - In scheduler/main.py calibration_check job (weekly, Monday 3 AM PT)
  - Directly via CLI for manual inspection

Usage:
  python -m backtesting.accuracy_tracker                  # Live DB summary
  python -m backtesting.accuracy_tracker --source csv --year 2024
  python -m backtesting.accuracy_tracker --window 30      # Rolling 30-day
  python -m backtesting.accuracy_tracker --alert          # Alert if Brier degrades
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import date, timedelta
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
# Data loading
# ---------------------------------------------------------------------------

def load_from_db(
    window_days: int = 365,
    min_games:   int = 5,
) -> List[Dict]:
    """
    Load prediction + outcome records from CalibrationHistory table.
    Only includes games where actual_outcome is not None (game has been played).

    Returns list of dicts: {game_date, predicted_prob, actual_outcome}
    """
    try:
        from data.cache.db import CalibrationHistory, get_session

        cutoff = date.today() - timedelta(days=window_days)

        with get_session() as session:
            rows = (
                session.query(CalibrationHistory)
                .filter(
                    CalibrationHistory.actual_outcome.isnot(None),
                    CalibrationHistory.game_date >= cutoff,
                )
                .order_by(CalibrationHistory.game_date)
                .all()
            )

        records = [
            {
                "game_pk":        row.game_pk,
                "game_date":      str(row.game_date),
                "predicted_prob": float(row.home_win_prob),
                "actual_outcome": int(row.actual_outcome),
            }
            for row in rows
            if row.actual_outcome is not None
        ]

        logger.debug("Loaded %d prediction records from DB.", len(records))
        return records

    except Exception as exc:
        logger.warning("Could not load from DB: %s", exc)
        return []


def load_from_csv(year: int) -> List[Dict]:
    """
    Load prediction + outcome records from backtest results CSV.
    Returns same format as load_from_db.
    """
    from config import HISTORICAL_DIR
    path = HISTORICAL_DIR / f"backtest_results_{year}.csv"

    if not path.exists():
        logger.warning("Backtest CSV not found: %s", path)
        return []

    records = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "game_pk":        row.get("game_pk"),
                    "game_date":      row["game_date"],
                    "predicted_prob": float(row["calibrated_prob"]),
                    "actual_outcome": int(row["home_win"]),
                })
            except (KeyError, ValueError):
                continue

    logger.debug("Loaded %d records from %s", len(records), path.name)
    return records


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _brier(probs: List[float], actuals: List[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs)


def _accuracy(probs: List[float], actuals: List[int]) -> float:
    if not probs:
        return 0.0
    return sum(int((p >= 0.5) == bool(a)) for p, a in zip(probs, actuals)) / len(probs)


def compute_rolling_metrics(
    records:     List[Dict],
    window_days: int = 30,
    as_of:       Optional[date] = None,
) -> Dict:
    """
    Compute rolling metrics over the last `window_days` days.

    Returns dict with:
      n_games, accuracy, brier_score, window_days, as_of_date
    """
    if as_of is None:
        as_of = date.today()

    cutoff = as_of - timedelta(days=window_days)
    window = [
        r for r in records
        if r["game_date"] >= str(cutoff) and r["game_date"] <= str(as_of)
    ]

    probs   = [r["predicted_prob"] for r in window]
    actuals = [r["actual_outcome"] for r in window]

    return {
        "n_games":      len(window),
        "accuracy":     round(_accuracy(probs, actuals), 4) if probs else 0.0,
        "brier_score":  round(_brier(probs, actuals), 4) if probs else 0.0,
        "window_days":  window_days,
        "as_of_date":   str(as_of),
    }


def compute_season_summary(records: List[Dict]) -> Dict:
    """Compute full-season accuracy and Brier score from all records."""
    probs   = [r["predicted_prob"] for r in records]
    actuals = [r["actual_outcome"] for r in records]

    wins_actual = sum(actuals)
    correct     = sum(int((p >= 0.5) == bool(a)) for p, a in zip(probs, actuals))

    return {
        "n_games":        len(records),
        "n_correct":      correct,
        "n_home_wins":    wins_actual,
        "accuracy":       round(_accuracy(probs, actuals), 4),
        "brier_score":    round(_brier(probs, actuals), 4),
        "home_win_rate":  round(wins_actual / len(actuals), 4) if actuals else 0.0,
    }


def compute_monthly_breakdown(records: List[Dict]) -> Dict[str, Dict]:
    """Break accuracy and Brier score down by month."""
    monthly: Dict[str, List] = defaultdict(list)
    for r in records:
        month = r["game_date"][:7]   # "YYYY-MM"
        monthly[month].append(r)

    result = {}
    for month, month_records in sorted(monthly.items()):
        probs   = [r["predicted_prob"] for r in month_records]
        actuals = [r["actual_outcome"] for r in month_records]
        result[month] = {
            "n_games":     len(month_records),
            "accuracy":    round(_accuracy(probs, actuals), 4),
            "brier_score": round(_brier(probs, actuals), 4),
        }

    return result


def detect_trend(records: List[Dict], window: int = 14) -> str:
    """
    Compare rolling accuracy over the last 14 games vs. the prior 14 games.
    Returns: "improving" | "degrading" | "stable" | "insufficient_data"
    """
    if len(records) < window * 2:
        return "insufficient_data"

    # Most recent window
    recent   = records[-window:]
    prior    = records[-(window * 2):-window]

    recent_acc = _accuracy(
        [r["predicted_prob"] for r in recent],
        [r["actual_outcome"] for r in recent],
    )
    prior_acc  = _accuracy(
        [r["predicted_prob"] for r in prior],
        [r["actual_outcome"] for r in prior],
    )

    delta = recent_acc - prior_acc

    if   delta >  0.03:  return "improving"
    elif delta < -0.03:  return "degrading"
    else:                return "stable"


def check_brier_alert(records: List[Dict], window_days: int = 30) -> Tuple[bool, float]:
    """
    Check if rolling Brier score exceeds the alert threshold.
    Returns (alert_triggered: bool, brier_score: float).
    """
    from config import BRIER_ALERT_THRESHOLD

    metrics = compute_rolling_metrics(records, window_days=window_days)
    brier   = metrics["brier_score"]
    n       = metrics["n_games"]

    if n < 10:
        return False, brier   # Not enough data to alert

    alert = brier > BRIER_ALERT_THRESHOLD
    return alert, brier


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def format_accuracy_report(
    records:     List[Dict],
    window_days: int = 30,
    source:      str = "database",
) -> str:
    """
    Format a multi-section accuracy report as a string.
    Used in report footer and scheduler logs.
    """
    lines = []
    lines.append("ACCURACY TRACKER")
    lines.append(f"Source: {source}  |  Total records: {len(records)}")
    lines.append("")

    if not records:
        lines.append("No prediction history available yet.")
        lines.append("Records accumulate as games are played daily.")
        return "\n".join(lines)

    # Rolling window
    rolling = compute_rolling_metrics(records, window_days=window_days)
    lines.append(f"ROLLING {window_days}-DAY WINDOW")
    lines.append(f"  Games:       {rolling['n_games']}")
    lines.append(f"  Accuracy:    {rolling['accuracy'] * 100:.1f}%   "
                 f"(Vegas benchmark: ~58.2%)")
    lines.append(f"  Brier Score: {rolling['brier_score']:.4f}   "
                 f"(target: < 0.240)")
    lines.append("")

    # Season summary
    season = compute_season_summary(records)
    lines.append("SEASON TO DATE")
    lines.append(f"  Games:       {season['n_games']}")
    lines.append(f"  Record:      {season['n_correct']}-{season['n_games'] - season['n_correct']}")
    lines.append(f"  Accuracy:    {season['accuracy'] * 100:.1f}%")
    lines.append(f"  Brier Score: {season['brier_score']:.4f}")
    lines.append(f"  Home win %:  {season['home_win_rate'] * 100:.1f}%")
    lines.append("")

    # Trend
    trend = detect_trend(records)
    trend_icon = {"improving": "▲", "degrading": "▼", "stable": "─", "insufficient_data": "?"}.get(trend, "?")
    lines.append(f"RECENT TREND ({window // 2 if (window := 14) else 14} game window)")
    lines.append(f"  Status: {trend_icon} {trend.replace('_', ' ').upper()}")
    lines.append("")

    # Monthly breakdown
    monthly = compute_monthly_breakdown(records)
    if monthly:
        lines.append("MONTHLY BREAKDOWN")
        for month, m in monthly.items():
            bar   = "█" * int(m["accuracy"] * 20)
            lines.append(
                f"  {month}:  {m['accuracy'] * 100:5.1f}%  "
                f"Brier={m['brier_score']:.4f}  "
                f"n={m['n_games']:4d}  |{bar:<20}|"
            )

    # Alert check
    alert, brier_val = check_brier_alert(records)
    if alert:
        from config import BRIER_ALERT_THRESHOLD
        lines.append("")
        lines.append(f"⚠ ALERT: Rolling Brier {brier_val:.4f} > threshold {BRIER_ALERT_THRESHOLD}")
        lines.append("  Investigate: recent model drift, data quality issue, or streak of upsets.")

    return "\n".join(lines)


def print_accuracy_report(
    records:     List[Dict],
    window_days: int = 30,
    source:      str = "database",
) -> None:
    """Print the accuracy report to logger."""
    report = format_accuracy_report(records, window_days, source)
    for line in report.split("\n"):
        logger.info(line)


# ---------------------------------------------------------------------------
# Report footer data (called by report/generator.py)
# ---------------------------------------------------------------------------

def get_report_footer_stats(window_days: int = 30) -> Dict:
    """
    Get the rolling accuracy stats for display in the daily report footer.
    Tries DB first, falls back to backtest CSV for current year.

    Returns dict suitable for Jinja2 template rendering.
    """
    records = load_from_db(window_days=max(window_days, 90))

    if not records:
        # Try current year backtest CSV
        current_year = date.today().year
        records = load_from_csv(current_year)
        if not records:
            # Try prior year
            records = load_from_csv(current_year - 1)

    if not records:
        return {
            "brier_score":        None,
            "accuracy":           None,
            "n_games":            0,
            "season_record":      "0-0",
            "calibration_status": "No data",
            "window_days":        window_days,
        }

    rolling = compute_rolling_metrics(records, window_days=window_days)
    season  = compute_season_summary(records)
    alert, brier_val = check_brier_alert(records)

    # Calibration method status
    try:
        from ensemble.calibrator import get_calibrator
        cal = get_calibrator()
        cal_status = f"{cal.method.title()} active (n={cal.n_games} games)"
    except Exception:
        cal_status = "Platt scaling (bootstrapping)"

    season_record = f"{season['n_correct']}-{season['n_games'] - season['n_correct']}"
    acc_pct       = f"{rolling['accuracy'] * 100:.1f}%"

    return {
        "brier_score":        rolling["brier_score"],
        "brier_score_fmt":    f"{rolling['brier_score']:.4f}" if rolling["brier_score"] else "—",
        "accuracy":           rolling["accuracy"],
        "accuracy_fmt":       acc_pct if rolling["n_games"] else "—",
        "n_games":            rolling["n_games"],
        "season_record":      season_record,
        "season_accuracy":    f"{season['accuracy'] * 100:.1f}%",
        "calibration_status": cal_status,
        "brier_alert":        alert,
        "window_days":        window_days,
        "trend":              detect_trend(records),
    }


# ---------------------------------------------------------------------------
# Log single prediction (called by morning_job after each game)
# ---------------------------------------------------------------------------

def log_prediction(
    game_pk:    int,
    game_date:  date,
    home_prob:  float,
    model_ver:  str = "v1.0",
    raw_prob:   Optional[float] = None,
) -> bool:
    """
    Store a new prediction in CalibrationHistory (actual_outcome filled in later).

    Returns True on success, False on failure.
    """
    try:
        from data.cache.db import CalibrationHistory, get_session

        with get_session() as session:
            existing = session.query(CalibrationHistory).filter(
                CalibrationHistory.game_pk == game_pk,
            ).first()

            if existing:
                existing.home_win_prob = home_prob
                existing.model_version = model_ver
                if raw_prob is not None:
                    existing.ensemble_raw = raw_prob
            else:
                row = CalibrationHistory(
                    game_pk       = game_pk,
                    game_date     = game_date,
                    home_win_prob = home_prob,
                    model_version = model_ver,
                    ensemble_raw  = raw_prob or home_prob,
                )
                session.add(row)

        return True

    except Exception as exc:
        logger.debug("Could not log prediction for game %s: %s", game_pk, exc)
        return False


def update_outcome(game_pk: int, home_won: bool) -> bool:
    """
    Fill in the actual_outcome for a previously logged prediction.
    Called by the nightly Elo update job.

    Returns True on success, False if record not found or DB error.
    """
    try:
        from data.cache.db import CalibrationHistory, get_session

        with get_session() as session:
            row = session.query(CalibrationHistory).filter(
                CalibrationHistory.game_pk == game_pk,
            ).first()

            if row:
                row.actual_outcome = 1 if home_won else 0
                return True
            else:
                logger.debug("No CalibrationHistory record found for game_pk=%s", game_pk)
                return False

    except Exception as exc:
        logger.debug("Could not update outcome for game %s: %s", game_pk, exc)
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Apex Analytics accuracy tracker")
    parser.add_argument(
        "--source", choices=["db", "csv"], default="db",
        help="Data source: 'db' for live predictions, 'csv' for backtest (default: db)",
    )
    parser.add_argument("--year",    type=int,  default=date.today().year, help="Season year (csv mode)")
    parser.add_argument("--window",  type=int,  default=30,   help="Rolling window in days (default: 30)")
    parser.add_argument("--alert",   action="store_true",     help="Exit with code 1 if Brier alert triggered")
    args = parser.parse_args()

    if args.source == "db":
        records = load_from_db(window_days=365)
        source_label = "CalibrationHistory DB"
    else:
        records = load_from_csv(args.year)
        source_label = f"backtest_results_{args.year}.csv"

    print_accuracy_report(records, window_days=args.window, source=source_label)

    if args.alert and records:
        triggered, brier = check_brier_alert(records, window_days=args.window)
        if triggered:
            logger.warning("ALERT: Brier %.4f exceeds threshold — exiting with code 1", brier)
            sys.exit(1)


if __name__ == "__main__":
    main()
