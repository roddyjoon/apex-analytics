"""
Apex Analytics — Bayesian Prior Blending System
Anchors early-season predictions to prior-season actuals.
Gradually releases weight toward current-season stats as sample size grows.

The hardest problem in baseball prediction is April.
A pitcher with 1 start has 15 BF. That's noise, not signal.
Solution: use prior-season ACTUALS as the prior — real data, not projection uncertainty.

Blend formula:
  in_season_weight = min(1.0, current_pa / FULL_SEASON_PA_THRESHOLD)
  blended = current_stat × in_season_weight + prior_stat × (1 - in_season_weight)

  At 30 PA  (early April):  5% current,  95% prior
  At 150 PA (mid-May):     25% current,  75% prior
  At 300 PA (mid-June):    50% current,  50% prior
  At 500 PA (August):      83% current,  17% prior
  At 600 PA (full season): 100% current,  0% prior
"""

import logging
from typing import Optional

from config import (
    BAYESIAN_FULL_SEASON_PA,
    BAYESIAN_FULL_SEASON_BF,
    LOW_CONFIDENCE_PA_THRESH,
    LOW_CONFIDENCE_BF_THRESH,
    LEAGUE_AVG_XWOBA,
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_K_RATE,
    LEAGUE_AVG_BB_RATE,
    LEAGUE_AVG_BARREL_PCT,
    LEAGUE_AVG_SWSTR_PCT,
)
from data.cache.db import get_prior_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core blending function
# ---------------------------------------------------------------------------


def blend_stat(
    current_val: Optional[float],
    prior_val:   Optional[float],
    sample_count: int,
    full_season_threshold: int,
) -> float:
    """
    Blend current-season stat with prior-season stat based on sample size.

    Parameters
    ----------
    current_val           : Current season stat value (can be None if no data yet).
    prior_val             : Prior season stat value (the Bayesian anchor).
    sample_count          : Current season PA (batters) or BF (pitchers).
    full_season_threshold : PA or BF count that gives 100% current weight.

    Returns
    -------
    Blended float value.
    """
    # Handle missing values
    if current_val is None and prior_val is None:
        return float("nan")
    if current_val is None:
        return float(prior_val)
    if prior_val is None:
        return float(current_val)

    in_season_weight = min(1.0, sample_count / full_season_threshold)
    prior_weight     = 1.0 - in_season_weight

    blended = (float(current_val) * in_season_weight) + (float(prior_val) * prior_weight)
    return blended


def blend_batter_stats(
    current_stats: dict,
    player_id: int,
    season: int,
) -> dict:
    """
    Apply Bayesian blending to all batter stats.
    Fetches prior season actuals from DB; falls back to league average if not found.

    Parameters
    ----------
    current_stats : Dict of current season stats from statcast_batter.py.
    player_id     : MLB Stats API person ID.
    season        : Current season year.

    Returns
    -------
    Dict with all stats blended; adds confidence and low_confidence_flag.
    """
    pa = current_stats.get("pa", 0)

    # Get prior season actuals from DB
    prior = _get_prior_batter(player_id, season)

    # Blend each key stat
    blended = dict(current_stats)  # Start with current stats as base
    blended["pa"] = pa

    stat_keys = ["xwoba", "xba", "barrel_pct", "hard_hit_pct",
                 "swstr_pct", "k_pct", "bb_pct", "hr_rate", "obp", "slg"]

    for key in stat_keys:
        current_val = current_stats.get(key)
        prior_val   = prior.get(key)
        blended[key] = blend_stat(
            current_val  = current_val,
            prior_val    = prior_val,
            sample_count = pa,
            full_season_threshold = BAYESIAN_FULL_SEASON_PA,
        )

    # Platoon splits — blend separately; fall back to overall xwOBA if split missing
    for split_key in ["xwoba_vs_lhp", "xwoba_vs_rhp"]:
        current_split = current_stats.get(split_key)
        prior_split   = prior.get(split_key)
        blended[split_key] = blend_stat(
            current_val  = current_split or blended["xwoba"],
            prior_val    = prior_split   or prior.get("xwoba"),
            sample_count = pa,
            full_season_threshold = BAYESIAN_FULL_SEASON_PA,
        )

    # Confidence flag
    blended["confidence"] = _batter_confidence(pa)
    blended["low_confidence_flag"] = pa < LOW_CONFIDENCE_PA_THRESH
    blended["in_season_weight"] = round(min(1.0, pa / BAYESIAN_FULL_SEASON_PA), 3)
    blended["prior_source"] = prior.get("_source", "league_avg")

    logger.debug(
        "Batter %d blend: PA=%d, in_season=%.2f, xwOBA %.3f→%.3f (prior=%.3f)",
        player_id, pa, blended["in_season_weight"],
        current_stats.get("xwoba", 0), blended["xwoba"],
        prior.get("xwoba", LEAGUE_AVG_XWOBA),
    )
    return blended


def blend_pitcher_stats(
    current_stats: dict,
    player_id: int,
    season: int,
) -> dict:
    """
    Apply Bayesian blending to all pitcher stats.
    """
    bf = current_stats.get("bf", 0)

    prior = _get_prior_pitcher(player_id, season)

    blended = dict(current_stats)
    blended["bf"] = bf

    # Core stats to blend
    stat_keys = [
        "xera", "k_pct", "bb_pct", "gb_pct", "fb_pct", "ld_pct",
        "swstr_pct", "csw_pct", "barrel_pct_allowed", "xba_allowed",
    ]

    for key in stat_keys:
        current_val = current_stats.get(key)
        prior_val   = prior.get(key)
        blended[key] = blend_stat(
            current_val  = current_val,
            prior_val    = prior_val,
            sample_count = bf,
            full_season_threshold = BAYESIAN_FULL_SEASON_BF,
        )

    # FIP components
    for key in ["fip_k_pct", "fip_bb_pct", "fip_hr_rate"]:
        blended[key] = blend_stat(
            current_val  = current_stats.get(key),
            prior_val    = prior.get(key),
            sample_count = bf,
            full_season_threshold = BAYESIAN_FULL_SEASON_BF,
        )

    blended["confidence"] = _pitcher_confidence(bf)
    blended["low_confidence_flag"] = bf < LOW_CONFIDENCE_BF_THRESH
    blended["in_season_weight"] = round(min(1.0, bf / BAYESIAN_FULL_SEASON_BF), 3)
    blended["prior_source"] = prior.get("_source", "league_avg")

    logger.debug(
        "Pitcher %d blend: BF=%d, in_season=%.2f, xERA %.3f→%.3f (prior=%.3f)",
        player_id, bf, blended["in_season_weight"],
        current_stats.get("xera", 0), blended["xera"],
        prior.get("xera", LEAGUE_AVG_ERA),
    )
    return blended


# ---------------------------------------------------------------------------
# Private: prior data retrieval with league-average fallback
# ---------------------------------------------------------------------------


def _get_prior_batter(player_id: int, current_season: int) -> dict:
    """
    Retrieve prior season actuals for a batter.
    Falls back to league average if no DB record.
    """
    prior_season = current_season - 1
    db_prior = get_prior_stats(player_id, prior_season, "batter")
    if db_prior:
        db_prior["_source"] = f"{prior_season}_actuals"
        return db_prior

    # Try 2-season average if prior season missing (injury-truncated)
    db_prior_2 = get_prior_stats(player_id, current_season - 2, "batter")
    if db_prior_2:
        db_prior_2["_source"] = f"{current_season - 2}_actuals"
        return db_prior_2

    logger.debug("No prior stats for batter %d — using league avg as prior.", player_id)
    return _league_avg_batter_prior()


def _get_prior_pitcher(player_id: int, current_season: int) -> dict:
    """
    Retrieve prior season actuals for a pitcher.
    Falls back to league average if no DB record.
    """
    prior_season = current_season - 1
    db_prior = get_prior_stats(player_id, prior_season, "pitcher")
    if db_prior:
        db_prior["_source"] = f"{prior_season}_actuals"
        return db_prior

    db_prior_2 = get_prior_stats(player_id, current_season - 2, "pitcher")
    if db_prior_2:
        db_prior_2["_source"] = f"{current_season - 2}_actuals"
        return db_prior_2

    logger.debug("No prior stats for pitcher %d — using league avg as prior.", player_id)
    return _league_avg_pitcher_prior()


def _league_avg_batter_prior() -> dict:
    """2024 MLB league-average batter stats as fallback prior."""
    return {
        "xwoba":         LEAGUE_AVG_XWOBA,
        "xba":           0.250,
        "barrel_pct":    LEAGUE_AVG_BARREL_PCT,
        "hard_hit_pct":  0.380,
        "swstr_pct":     LEAGUE_AVG_SWSTR_PCT,
        "k_pct":         LEAGUE_AVG_K_RATE,
        "bb_pct":        LEAGUE_AVG_BB_RATE,
        "hr_rate":       0.037,
        "obp":           0.315,
        "slg":           0.413,
        "xwoba_vs_lhp":  LEAGUE_AVG_XWOBA,
        "xwoba_vs_rhp":  LEAGUE_AVG_XWOBA,
        "_source":       "league_avg",
    }


def _league_avg_pitcher_prior() -> dict:
    """2024 MLB league-average pitcher stats as fallback prior."""
    return {
        "xera":               LEAGUE_AVG_ERA,
        "k_pct":              LEAGUE_AVG_K_RATE,
        "bb_pct":             LEAGUE_AVG_BB_RATE,
        "gb_pct":             0.430,
        "fb_pct":             0.350,
        "ld_pct":             0.220,
        "swstr_pct":          LEAGUE_AVG_SWSTR_PCT,
        "csw_pct":            0.280,
        "barrel_pct_allowed": LEAGUE_AVG_BARREL_PCT,
        "xba_allowed":        0.250,
        "fip_k_pct":          LEAGUE_AVG_K_RATE,
        "fip_bb_pct":         LEAGUE_AVG_BB_RATE,
        "fip_hr_rate":        0.037,
        "_source":            "league_avg",
    }


# ---------------------------------------------------------------------------
# Confidence tier helpers
# ---------------------------------------------------------------------------


def _batter_confidence(pa: int) -> str:
    if pa >= 100:
        return "HIGH"
    if pa >= LOW_CONFIDENCE_PA_THRESH:
        return "MEDIUM"
    return "LOW"


def _pitcher_confidence(bf: int) -> str:
    if bf >= 100:
        return "HIGH"
    if bf >= LOW_CONFIDENCE_BF_THRESH:
        return "MEDIUM"
    return "LOW"


def get_in_season_weight(sample_count: int, full_season_threshold: int) -> float:
    """Public helper: return the current-season weight (0.0–1.0) for a given sample size."""
    return round(min(1.0, sample_count / full_season_threshold), 3)
