"""
Apex Analytics — Dynamic Pitcher Removal Model
Probabilistic starter removal — NOT a flat 6-inning cutoff.

Average MLB starter 2024: 5.2 innings. The model tracks estimated pitch count
within each simulated game and removes starters dynamically using a
manager-decision probability model calibrated to real removal patterns.

Hard limits (always remove):
  - Estimated pitch count >= 100
  - Inning > 7 (never go past the 7th)
  - Runs allowed >= 5 (hook after blowup)

Probabilistic removal by inning (manager discretion model):
  After inning 4: 5%  chance
  After inning 5: 20% chance
  After inning 6: 55% chance
  After inning 7: 90% chance

Adjustments:
  + Fatigue > 0.6: ×1.3 removal probability
  + Runs allowed >= 3: ×1.4 removal probability
  + Stuff+ proxy > 110 (ace): ×0.8 removal probability (stays longer)
"""

import logging

import numpy as np

from config import (
    STARTER_REMOVAL_PROBS,
    MAX_STARTER_INNINGS,
    PITCHER_PITCH_COUNT_LIMIT,
    PITCHES_PER_INNING_BASE,
    PITCHES_PER_INNING_SLOPE,
    FATIGUE_PER_INNING,
    MAX_FATIGUE_PENALTY,
    LEAGUE_AVG_SWSTR_PCT,
)
from simulation.profiles import PitcherProfile

logger = logging.getLogger(__name__)


def should_remove_starter(
    pitcher:               PitcherProfile,
    inning:                int,
    runs_allowed:          int,
    rng:                   np.random.Generator,
) -> bool:
    """
    Determine whether the starter should be removed after completing this inning.

    Parameters
    ----------
    pitcher      : PitcherProfile with fatigue_index and pitch_count_est updated.
    inning       : Inning just completed (1-indexed: 1 = after 1st inning).
    runs_allowed : Total runs allowed by starter so far in this game.
    rng          : numpy random Generator.

    Returns
    -------
    True if starter should be pulled; False if he continues.

    Side effects
    ------------
    Updates pitcher.pitch_count_est and pitcher.fatigue_index each call.
    """
    # ── Update pitch count estimate ───────────────────────────────────────────
    pitches_this_inning = estimate_pitches_per_inning(inning)
    pitcher.pitch_count_est += pitches_this_inning

    # ── Update fatigue index ──────────────────────────────────────────────────
    pitcher.fatigue_index = min(
        MAX_FATIGUE_PENALTY,
        pitcher.fatigue_index + FATIGUE_PER_INNING,
    )

    # ── Hard limits — always remove ───────────────────────────────────────────
    if pitcher.pitch_count_est >= PITCHER_PITCH_COUNT_LIMIT:
        logger.debug("SP removed: pitch count %d >= %d limit.",
                     pitcher.pitch_count_est, PITCHER_PITCH_COUNT_LIMIT)
        return True

    if inning >= MAX_STARTER_INNINGS:
        logger.debug("SP removed: inning %d >= max %d.", inning, MAX_STARTER_INNINGS)
        return True

    if runs_allowed >= 5:
        logger.debug("SP removed: runs allowed %d >= 5 (hook threshold).", runs_allowed)
        return True

    # ── Probabilistic removal ─────────────────────────────────────────────────
    base_prob = STARTER_REMOVAL_PROBS.get(inning, 0.0)

    if base_prob <= 0.0:
        return False  # Innings 1-3: never remove probabilistically

    # Adjustments
    adjusted_prob = base_prob

    # Fatigue: tired pitcher gets pulled sooner
    if pitcher.fatigue_index > 0.06:  # ~4 innings pitched
        adjusted_prob *= 1.3

    # Struggling: runs allowed
    if runs_allowed >= 3:
        adjusted_prob *= 1.4
    elif runs_allowed >= 2:
        adjusted_prob *= 1.15

    # Ace stays longer: above-average stuff
    if pitcher.stuff_plus_proxy > 110:
        adjusted_prob *= 0.80
    elif pitcher.stuff_plus_proxy > 120:  # True ace
        adjusted_prob *= 0.65

    # Days rest: well-rested starters go deeper
    if (pitcher.days_rest or 0) >= 6:
        adjusted_prob *= 0.85

    # Cap at 1.0
    adjusted_prob = min(1.0, adjusted_prob)

    remove = rng.random() < adjusted_prob
    logger.debug(
        "Starter removal check [inning=%d, runs=%d, fatigue=%.3f, pitch_est=%d]: "
        "base_prob=%.2f → adj_prob=%.2f → remove=%s",
        inning, runs_allowed, pitcher.fatigue_index,
        pitcher.pitch_count_est, base_prob, adjusted_prob, remove
    )
    return remove


def estimate_pitches_per_inning(inning: int) -> int:
    """
    Estimate pitches thrown in a given inning number.
    Formula: 15 + (inning × 0.5) — pitchers throw slightly more as fatigue builds.

    Returns integer pitch count estimate for this inning.
    """
    return int(PITCHES_PER_INNING_BASE + (inning * PITCHES_PER_INNING_SLOPE))


def apply_fatigue_to_era(pitcher: PitcherProfile) -> float:
    """
    Return the pitcher's ERA adjusted for current fatigue state.
    Used in PA calculator when starter is still in the game but tired.

    fatigue_index → ERA multiplier (0 = no penalty, MAX_FATIGUE_PENALTY = 15% max)
    """
    penalty = min(pitcher.fatigue_index, MAX_FATIGUE_PENALTY)
    return pitcher.true_talent_era * (1.0 + penalty)


def reset_pitcher_fatigue(pitcher: PitcherProfile) -> None:
    """Reset fatigue and pitch count for a new simulation iteration."""
    pitcher.fatigue_index    = 0.0
    pitcher.pitch_count_est  = 0
