"""
Apex Analytics — Batter-vs-Pitcher Matchup History Adjuster
Applies head-to-head career PA history when sufficient sample exists.

Weight schedule (from config):
  10–29 PA : 20% matchup history, 80% model estimate
  30–74 PA : 35% matchup history, 65% model estimate
  75+   PA : 50% matchup history, 50% model estimate

Impact: Real head-to-head data can reveal mismatches that neither platoon
splits nor pitch-type vulnerability captures (e.g., batter who simply
"owns" a pitcher regardless of handedness or arsenal).
"""

import logging
from typing import Optional

from config import (
    MATCHUP_MIN_PA,
    MATCHUP_WEIGHT_SMALL,
    MATCHUP_WEIGHT_MEDIUM,
    MATCHUP_WEIGHT_LARGE,
    LEAGUE_AVG_XWOBA,
)
from data.cache.db import get_matchup_history

logger = logging.getLogger(__name__)


def get_matchup_xwoba_multiplier(
    batter_id:    int,
    pitcher_id:   int,
    batter_xwoba: float,
) -> tuple[float, str]:
    """
    Compute a multiplier applied to batter's base xwOBA using head-to-head history.

    Parameters
    ----------
    batter_id    : MLB Stats API person ID for the batter.
    pitcher_id   : MLB Stats API person ID for the pitcher.
    batter_xwoba : The batter's current (Bayesian-blended) xwOBA — used as the
                   base to blend against matchup xwOBA.

    Returns
    -------
    (multiplier, note_str)
    multiplier: float — applied to batter_xwoba in PA calculator (1.0 = no adjustment).
    note_str  : Human-readable note for report display, empty if no adjustment.
    """
    history = get_matchup_history(batter_id, pitcher_id)

    if history is None:
        return 1.0, ""

    n_pa = history.get("pa", 0)
    if n_pa < MATCHUP_MIN_PA:
        return 1.0, ""

    matchup_xwoba = history.get("xwoba")
    if matchup_xwoba is None:
        return 1.0, ""

    # Determine blend weight based on sample size
    matchup_weight = _get_matchup_weight(n_pa)
    model_weight   = 1.0 - matchup_weight

    # Blend matchup xwOBA with model's batter xwOBA
    blended_xwoba = (matchup_xwoba * matchup_weight) + (batter_xwoba * model_weight)

    # Convert to multiplier vs. batter's own model estimate
    if batter_xwoba <= 0:
        return 1.0, ""
    multiplier = blended_xwoba / batter_xwoba

    # Clamp to prevent outlier single-game sample effects
    multiplier = max(0.60, min(1.60, multiplier))

    # Build report note if adjustment is meaningful
    note = ""
    if abs(multiplier - 1.0) >= 0.03:
        direction = "advantage" if multiplier > 1.0 else "disadvantage"
        note = (
            f"Head-to-head history ({n_pa} PA): batter xwOBA .{int(matchup_xwoba*1000):03d} "
            f"({direction}, {matchup_weight*100:.0f}% weight)"
        )

    logger.debug(
        "Matchup adj [batter %d vs pitcher %d]: n=%d PA, matchup_xwoba=%.3f, "
        "model_xwoba=%.3f → blended=%.3f (mult=%.3f)",
        batter_id, pitcher_id, n_pa, matchup_xwoba, batter_xwoba, blended_xwoba, multiplier
    )
    return multiplier, note


def get_matchup_summary(batter_id: int, pitcher_id: int) -> Optional[dict]:
    """
    Return full matchup stats for report display table.
    Returns None if no history or below minimum PA.
    """
    history = get_matchup_history(batter_id, pitcher_id)
    if history is None:
        return None
    n_pa = history.get("pa", 0)
    if n_pa < MATCHUP_MIN_PA:
        return None
    return {
        "pa":     n_pa,
        "hits":   history.get("hits", 0),
        "hr":     history.get("hr", 0),
        "bb":     history.get("bb", 0),
        "k":      history.get("k", 0),
        "xwoba":  history.get("xwoba"),
        "weight": _get_matchup_weight(n_pa),
        "ba":     round(history.get("hits", 0) / max(history.get("ab", 1), 1), 3),
    }


def build_lineup_matchup_map(
    batter_ids:  list[int],
    pitcher_id:  int,
) -> dict[int, Optional[dict]]:
    """
    Build a map of {batter_id: matchup_summary} for an entire lineup.
    Used by the report generator to annotate lineup tables.
    """
    return {
        bid: get_matchup_summary(bid, pitcher_id)
        for bid in batter_ids
    }


def _get_matchup_weight(n_pa: int) -> float:
    """Return the matchup history blend weight for a given sample size."""
    if n_pa >= 75:
        return MATCHUP_WEIGHT_LARGE    # 0.50
    if n_pa >= 30:
        return MATCHUP_WEIGHT_MEDIUM   # 0.35
    if n_pa >= MATCHUP_MIN_PA:
        return MATCHUP_WEIGHT_SMALL    # 0.20
    return 0.0
