"""
Apex Analytics — Base Running Model
Sprint-speed-based base advancement tables for the Monte Carlo simulation.

Speed tiers (from config):
  Fast    : >= 28.0 ft/s  (top ~15% of MLB)
  Average : 25.0–27.9 ft/s
  Slow    : < 25.0 ft/s   (bottom ~15% of MLB)

Base state representation: tuple (on_1b: bool, on_2b: bool, on_3b: bool)
  (False, False, False) = bases empty
  (True,  False, False) = runner on 1B
  (True,  True,  True)  = bases loaded

Impact on run scoring: ~0.3–0.5 runs/game vs. simplified fixed-advance model.
"""

import logging
from typing import Optional

import numpy as np

from config import SPRINT_FAST_THRESHOLD, SPRINT_SLOW_THRESHOLD

logger = logging.getLogger(__name__)

# Base state type alias
BasesState = tuple[bool, bool, bool]  # (on_1b, on_2b, on_3b)

BASES_EMPTY: BasesState = (False, False, False)

# ---------------------------------------------------------------------------
# Advancement probability tables
# Format: {(on_1b, on_2b, on_3b): {outcome: (runs_scored, new_base_state)}}
# Probabilities vary by speed tier — these are the "average" tier base tables.
# Fast/slow tiers adjust specific probabilities.
# ---------------------------------------------------------------------------

# Single advancement: does runner on 1B go to 3B (vs stop at 2B)?
# Does runner on 2B score (vs stop at 3B)?
_SINGLE_ADVANCE_PROBS = {
    # (from_base): {speed: P(takes extra base)}
    "1b_to_3b": {"fast": 0.55, "average": 0.35, "slow": 0.15},
    "2b_scores": {"fast": 0.95, "average": 0.85, "slow": 0.70},
    "3b_scores": {"fast": 1.00, "average": 1.00, "slow": 1.00},
}

# Double advancement: does runner on 1B score (vs stop at 3B)?
_DOUBLE_ADVANCE_PROBS = {
    "1b_scores": {"fast": 0.85, "average": 0.65, "slow": 0.40},
    "2b_scores": {"fast": 1.00, "average": 1.00, "slow": 0.95},  # Almost always scores
    "3b_scores": {"fast": 1.00, "average": 1.00, "slow": 1.00},
}

# Sacrifice fly: runner on 3B tags and scores on flyout (< 2 outs)
_SAC_FLY_PROB = {"fast": 0.92, "average": 0.85, "slow": 0.75}

# Double play: probability when runner on 1B, < 2 outs, groundout
_DOUBLE_PLAY_PROB = 0.50  # ~50% of eligible situations result in DP

# Groundout with runner on 1B (<2 outs): runner advances to 2B on force
# (non-DP groundouts advance lead runners one base)


def _speed_tier(sprint_speed: float) -> str:
    if sprint_speed >= SPRINT_FAST_THRESHOLD:
        return "fast"
    if sprint_speed <= SPRINT_SLOW_THRESHOLD:
        return "slow"
    return "average"


def advance_runners(
    bases:         BasesState,
    outcome:       str,
    batter_speed:  float,
    rng:           np.random.Generator,
    outs:          int = 0,
) -> tuple[int, BasesState, int]:
    """
    Advance runners based on PA outcome and batter sprint speed.

    Parameters
    ----------
    bases        : Current base state (on_1b, on_2b, on_3b).
    outcome      : PA outcome string from pa_calculator.
    batter_speed : Batter sprint speed in ft/s (from BatterProfile.sprint_speed).
    rng          : numpy random Generator for stochastic advancement.
    outs         : Current outs before this PA (0, 1, or 2).

    Returns
    -------
    (runs_scored, new_bases_state, outs_added)
    runs_scored     : Runs scored on this PA.
    new_bases_state : Updated (on_1b, on_2b, on_3b) after advancement.
    outs_added      : Extra outs added (0 normally, 1 for DP, 1 for flyout/sac).
    """
    on_1b, on_2b, on_3b = bases
    speed = _speed_tier(batter_speed)

    if outcome == "hr":
        return _handle_hr(on_1b, on_2b, on_3b)

    if outcome == "triple":
        return _handle_triple(on_1b, on_2b, on_3b)

    if outcome == "double":
        return _handle_double(on_1b, on_2b, on_3b, speed, rng)

    if outcome == "single":
        return _handle_single(on_1b, on_2b, on_3b, speed, rng)

    if outcome in ("walk", "hbp"):
        return _handle_walk(on_1b, on_2b, on_3b)

    if outcome == "strikeout":
        return _handle_strikeout(on_1b, on_2b, on_3b)

    if outcome == "groundout":
        return _handle_groundout(on_1b, on_2b, on_3b, outs, rng)

    if outcome == "flyout":
        return _handle_flyout(on_1b, on_2b, on_3b, outs, speed, rng)

    # Unknown outcome — treat as out, no advancement
    logger.warning("Unknown outcome '%s' in base_running — treating as out.", outcome)
    return 0, bases, 1


# ---------------------------------------------------------------------------
# Per-outcome handlers
# ---------------------------------------------------------------------------

def _handle_hr(on_1b, on_2b, on_3b) -> tuple[int, BasesState, int]:
    """Home run: batter + all runners score. Bases cleared."""
    runners_scoring = int(on_1b) + int(on_2b) + int(on_3b) + 1  # +1 for batter
    return runners_scoring, BASES_EMPTY, 0


def _handle_triple(on_1b, on_2b, on_3b) -> tuple[int, BasesState, int]:
    """Triple: all runners score, batter on 3B."""
    runs = int(on_1b) + int(on_2b) + int(on_3b)
    return runs, (False, False, True), 0


def _handle_double(on_1b, on_2b, on_3b, speed, rng) -> tuple[int, BasesState, int]:
    """Double: complex — runner on 1B may or may not score."""
    runs = 0
    new_1b, new_2b, new_3b = False, False, False

    # Runner on 3B always scores on double
    if on_3b:
        runs += 1

    # Runner on 2B almost always scores
    if on_2b:
        p = _DOUBLE_ADVANCE_PROBS["2b_scores"][speed]
        if rng.random() < p:
            runs += 1
        else:
            new_3b = True  # Held at 3B (rare for fast runner)

    # Runner on 1B: scores (fast runner) or stops at 3B
    if on_1b:
        p = _DOUBLE_ADVANCE_PROBS["1b_scores"][speed]
        if rng.random() < p:
            runs += 1
        else:
            new_3b = True  # Stops at 3B (1B runner didn't score)

    # Batter on 2B
    new_2b = True

    return runs, (new_1b, new_2b, new_3b), 0


def _handle_single(on_1b, on_2b, on_3b, speed, rng) -> tuple[int, BasesState, int]:
    """Single: complex advancement based on speed."""
    runs = 0
    new_1b, new_2b, new_3b = False, False, False

    # Runner on 3B always scores on single
    if on_3b:
        runs += 1

    # Runner on 2B: scores or stops at 3B
    if on_2b:
        p = _SINGLE_ADVANCE_PROBS["2b_scores"][speed]
        if rng.random() < p:
            runs += 1
        else:
            new_3b = True

    # Runner on 1B: goes to 3B (fast/aggressive) or stops at 2B
    if on_1b:
        p = _SINGLE_ADVANCE_PROBS["1b_to_3b"][speed]
        if rng.random() < p:
            new_3b = True  # Took the extra base to 3B
        else:
            new_2b = True  # Stopped at 2B

    # Batter on 1B
    new_1b = True

    return runs, (new_1b, new_2b, new_3b), 0


def _handle_walk(on_1b, on_2b, on_3b) -> tuple[int, BasesState, int]:
    """
    Walk / HBP: advance runners only on force.
    Forced advancement is deterministic (no speed involved).
    """
    if on_1b and on_2b and on_3b:
        # Bases loaded: everyone forced — 3B runner scores
        return 1, (True, True, True), 0
    if on_1b and on_2b:
        # 1B and 2B occupied: all force up (3B runner doesn't exist → no score)
        return 0, (True, True, True), 0
    if on_1b:
        # Only 1B occupied: force to 2B
        return 0, (True, True, on_3b), 0
    # No one on 1B: no force, batter takes 1B
    return 0, (True, on_2b, on_3b), 0


def _handle_strikeout(on_1b, on_2b, on_3b) -> tuple[int, BasesState, int]:
    """Strikeout: no advancement, 1 out. (Stolen base on K-WP not modeled.)"""
    return 0, (on_1b, on_2b, on_3b), 1


def _handle_groundout(on_1b, on_2b, on_3b, outs, rng) -> tuple[int, BasesState, int]:
    """
    Groundout: potential double play when runner on 1B and < 2 outs.
    Non-DP groundout: lead runner advances one base on force.
    """
    runs = 0
    new_1b, new_2b, new_3b = on_1b, on_2b, on_3b
    outs_added = 1

    # Double play scenario: runner on 1B, less than 2 outs
    if on_1b and outs < 2:
        if rng.random() < _DOUBLE_PLAY_PROB:
            # DP: batter out at 1B, runner out at 2B
            outs_added = 2
            new_1b = False
            # Other runners advance one base on the play (runner from 2B to 3B, etc.)
            if on_2b and not on_3b:
                new_3b = True
                new_2b = False
            return runs, (new_1b, new_2b, new_3b), outs_added

    # Non-DP groundout: batter out, lead runners advance on force if applicable
    new_1b = False  # Batter out at 1B

    # Force advancement when bases were occupied consecutively
    if on_1b and on_2b and on_3b:
        # Bases loaded: force at every base; runner on 3B scores
        runs += 1
        new_1b = True   # Runner from 1B forced to 2B? No — batter is out
        # Actually: batter out at 1B on grounder; runners forced up
        # Runner from 1B goes to 2B, runner from 2B goes to 3B, runner from 3B scores
        new_1b = False
        new_2b = True   # Runner from 1B
        new_3b = True   # Runner from 2B (was already at 3B, now scores -> runner from 2B takes 3B)
        # Simplify: loaded base grounder = runner scores, others advance, batter out
        return runs, (False, True, True), outs_added

    if on_1b and on_2b:
        # Force at 2B and 3B; batter out
        new_2b = False
        new_3b = True   # Runner from 2B
        new_1b = False  # But runner from 1B goes to 2B... wait
        # Correction: batter is out at 1B; runner from 1B goes to 2B (forced);
        # runner from 2B goes to 3B (forced)
        return runs, (False, True, True), outs_added

    if on_1b:
        # Runner forced from 1B to 2B; batter out at 1B
        # Result: runner now on 2B
        return runs, (False, True, on_3b), outs_added

    # No runner on 1B: no force; batter simply out
    # Runner on 3B: may try to score on groundout (sacrifice-groundout) — rare, skip
    return runs, (False, on_2b, on_3b), outs_added


def _handle_flyout(on_1b, on_2b, on_3b, outs, speed, rng) -> tuple[int, BasesState, int]:
    """
    Flyout: sacrifice fly possibility when runner on 3B and < 2 outs.
    Other runners may tag and advance (modeled for 3B→score only).
    """
    runs = 0
    new_1b, new_2b, new_3b = on_1b, on_2b, on_3b
    outs_added = 1

    # Sacrifice fly: runner on 3B can tag and score if < 2 outs
    if on_3b and outs < 2:
        p = _SAC_FLY_PROB[speed]
        if rng.random() < p:
            runs += 1
            new_3b = False

    return runs, (new_1b, new_2b, new_3b), outs_added


# ---------------------------------------------------------------------------
# Helper: count runners on base
# ---------------------------------------------------------------------------

def count_runners(bases: BasesState) -> int:
    return sum(bases)


def bases_description(bases: BasesState) -> str:
    """Human-readable base state for logging."""
    on_1b, on_2b, on_3b = bases
    occupied = []
    if on_1b: occupied.append("1B")
    if on_2b: occupied.append("2B")
    if on_3b: occupied.append("3B")
    return "+".join(occupied) if occupied else "empty"
