"""
Apex Analytics — Exponential Decay Recency Weighter
Weights recent games more heavily than older games using exponential decay.

Why exponential decay vs. flat rolling window:
  A dominant 3-start run 5 days ago should matter MORE than a clunker 18 days ago.
  Flat windows treat all games within the window identically — that's wrong.
  Exponential decay gives each game a weight proportional to 0.92^days_ago.

  game from today:   0.92^0  = 1.00 weight
  game from 5 days:  0.92^5  = 0.66 weight
  game from 10 days: 0.92^10 = 0.43 weight
  game from 21 days: 0.92^21 = 0.18 weight  (near-irrelevant)
  game from 30 days: 0.92^30 = 0.08 weight  (almost zero)
"""

import logging
from datetime import date, timedelta
from typing import Optional

from config import (
    DECAY_FACTOR,
    DECAY_MIN_GAMES,
    DECAY_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)


def exponential_weighted_stat(
    game_log: list[dict],
    stat_field: str,
    reference_date: Optional[date] = None,
    window_days: int = DECAY_WINDOW_DAYS,
    min_games: int = DECAY_MIN_GAMES,
    decay_factor: float = DECAY_FACTOR,
) -> Optional[float]:
    """
    Compute exponentially-weighted average of a stat from a game log.

    Parameters
    ----------
    game_log      : List of dicts, each with a "date" key (ISO str) and stat_field value.
                    Sorted order doesn't matter; we sort by date internally.
    stat_field    : The stat to average (e.g. "xwoba", "xera", "k_pct").
    reference_date: Date to compute weights relative to (defaults to today).
    window_days   : Only include games within this many days of reference_date.
    min_games     : Minimum observations required before returning a result.
                    Returns None if fewer games available (caller uses season average).
    decay_factor  : Per-day decay rate (default 0.92 → 8% decay per day).

    Returns
    -------
    Exponentially-weighted average, or None if insufficient data.
    """
    if not game_log:
        return None

    if reference_date is None:
        reference_date = date.today()

    cutoff = reference_date - timedelta(days=window_days)

    weighted_sum   = 0.0
    total_weight   = 0.0
    valid_count    = 0

    for entry in game_log:
        # Parse date
        entry_date = _parse_date(entry.get("date"))
        if entry_date is None or entry_date < cutoff or entry_date > reference_date:
            continue

        # Get stat value
        val = entry.get(stat_field)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if val != val:  # NaN check
            continue

        days_ago = (reference_date - entry_date).days
        weight   = decay_factor ** days_ago

        weighted_sum  += val * weight
        total_weight  += weight
        valid_count   += 1

    if valid_count < min_games or total_weight <= 0:
        logger.debug(
            "Insufficient decay data: %d games for '%s' (need %d). Returning None.",
            valid_count, stat_field, min_games
        )
        return None

    result = weighted_sum / total_weight
    logger.debug(
        "Decay stat '%s': %.4f  (n=%d games, window=%d days)",
        stat_field, result, valid_count, window_days
    )
    return result


def compute_batter_recent_xwoba(
    game_log: list[dict],
    reference_date: Optional[date] = None,
) -> Optional[float]:
    """
    Convenience: compute exponentially-weighted xwOBA for a batter's recent games.
    Returns None if fewer than DECAY_MIN_GAMES observations in window.
    """
    return exponential_weighted_stat(
        game_log=game_log,
        stat_field="xwoba",
        reference_date=reference_date,
    )


def compute_pitcher_recent_xera(
    start_log: list[dict],
    reference_date: Optional[date] = None,
    n_starts: int = 5,
) -> Optional[float]:
    """
    Compute exponentially-weighted xERA over last N starts.
    Uses the same decay formula but limited to recent starts only.
    """
    if not start_log:
        return None

    # Sort by date descending and take last N starts
    sorted_log = sorted(start_log, key=lambda x: x.get("date", ""), reverse=True)
    recent = sorted_log[:n_starts]

    return exponential_weighted_stat(
        game_log=recent,
        stat_field="xera",
        reference_date=reference_date,
        window_days=60,           # Wider window for pitchers (every 5 days)
        min_games=2,              # Lower threshold: 2 recent starts is enough signal
    )


def compute_pitcher_recent_k_pct(
    start_log: list[dict],
    reference_date: Optional[date] = None,
    n_starts: int = 5,
) -> Optional[float]:
    sorted_log = sorted(start_log, key=lambda x: x.get("date", ""), reverse=True)
    recent = sorted_log[:n_starts]
    return exponential_weighted_stat(
        game_log=recent,
        stat_field="k_pct",
        reference_date=reference_date,
        window_days=60,
        min_games=2,
    )


def compute_pitcher_recent_bb_pct(
    start_log: list[dict],
    reference_date: Optional[date] = None,
    n_starts: int = 5,
) -> Optional[float]:
    sorted_log = sorted(start_log, key=lambda x: x.get("date", ""), reverse=True)
    recent = sorted_log[:n_starts]
    return exponential_weighted_stat(
        game_log=recent,
        stat_field="bb_pct",
        reference_date=reference_date,
        window_days=60,
        min_games=2,
    )


def compute_team_recent_win_pct(
    game_results: list[dict],
    reference_date: Optional[date] = None,
    window_games: int = 15,
) -> Optional[float]:
    """
    Compute exponentially-weighted win% for a team over recent games.
    Used as a feature in the Random Forest and Logistic Regression models.

    game_results: list of {date: str, won: bool (1 or 0)} dicts.
    """
    return exponential_weighted_stat(
        game_log=game_results,
        stat_field="won",
        reference_date=reference_date,
        window_days=window_games * 2,   # Rough day equivalent
        min_games=5,
    )


def get_decay_weights_for_log(
    game_log: list[dict],
    reference_date: Optional[date] = None,
    window_days: int = DECAY_WINDOW_DAYS,
) -> list[tuple[dict, float]]:
    """
    Return (game_entry, weight) pairs for diagnostic / explainability use.
    Useful for showing "why is the model rating this pitcher as hot/cold".
    """
    if reference_date is None:
        reference_date = date.today()

    cutoff = reference_date - timedelta(days=window_days)
    result = []

    for entry in game_log:
        entry_date = _parse_date(entry.get("date"))
        if entry_date is None or entry_date < cutoff or entry_date > reference_date:
            continue
        days_ago = (reference_date - entry_date).days
        weight   = DECAY_FACTOR ** days_ago
        result.append((entry, round(weight, 4)))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date(date_val) -> Optional[date]:
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, str):
        try:
            return date.fromisoformat(date_val[:10])
        except ValueError:
            return None
    return None
