"""
Apex Analytics — Half-Inning Simulator
Simulates one half-inning using a Markov base-state loop.

Each plate appearance:
  1. Compute PA outcome probabilities (pa_calculator)
  2. Sample one outcome
  3. Advance runners (base_running)
  4. Track outs, runs, base state

Returns: (runs_scored, next_lineup_index)
The next_lineup_index carries over so the leadoff batter next inning
is whoever was due up when the 3rd out was recorded.
"""

import logging

import numpy as np

from simulation.profiles import BatterProfile, PitcherProfile, ParkContext
from simulation.pa_calculator import compute_pa_outcomes, sample_outcome, is_out
from simulation.base_running import (
    BasesState, BASES_EMPTY, advance_runners, bases_description
)
from simulation.pitcher_removal import apply_fatigue_to_era

logger = logging.getLogger(__name__)

MAX_PA_PER_INNING = 27  # Safety cap: 9 batters × 3 base hits each — prevent infinite loops


def simulate_half_inning(
    lineup:      list[BatterProfile],
    lineup_idx:  int,
    pitcher:     PitcherProfile,
    park:        ParkContext,
    rng:         np.random.Generator,
    ghost_runner: bool = False,
) -> tuple[int, int]:
    """
    Simulate one half-inning (3 outs).

    Parameters
    ----------
    lineup      : List of BatterProfiles in batting order (9 slots, may wrap).
    lineup_idx  : Index of the leadoff batter for this half-inning (0-8).
    pitcher     : PitcherProfile of the current pitcher (starter or bullpen).
    park        : ParkContext with park factors and weather.
    rng         : numpy random Generator.
    ghost_runner: If True, start with runner on 2B (extra innings MLB rule).

    Returns
    -------
    (runs_scored, next_lineup_idx)
    runs_scored    : Total runs scored this half-inning.
    next_lineup_idx: Batting order index of the next batter after 3rd out.
    """
    outs  = 0
    runs  = 0
    bases: BasesState = BASES_EMPTY
    pa_count = 0

    # Extra innings ghost runner: starts on 2B
    if ghost_runner:
        bases = (False, True, False)
        logger.debug("Ghost runner placed on 2B for extra innings.")

    while outs < 3:
        if pa_count >= MAX_PA_PER_INNING:
            logger.warning("PA safety cap (%d) hit — ending inning early.", MAX_PA_PER_INNING)
            break

        # Get current batter (wrap lineup 0-8)
        batter = lineup[lineup_idx % 9]

        # Apply fatigue to pitcher's ERA in PA calculator
        # (fatigue is tracked on the pitcher object; pa_calculator reads it)
        # The fatigue ERA adjustment is built into pa_calculator via pitcher.fatigue_index

        # Compute PA probabilities
        probs = compute_pa_outcomes(
            batter=batter,
            pitcher=pitcher,
            park=park,
            outs=outs,
        )

        # Sample outcome
        outcome = sample_outcome(probs, rng)

        # Advance runners
        runs_scored, new_bases, outs_added = advance_runners(
            bases=bases,
            outcome=outcome,
            batter_speed=batter.sprint_speed,
            rng=rng,
            outs=outs,
        )

        # Update state
        runs  += runs_scored
        bases  = new_bases
        outs  += outs_added

        logger.debug(
            "  PA: %s vs %s → %s | runners: %s → %s | runs: +%d | outs: %d",
            batter.player_name[:12], pitcher.player_name[:12],
            outcome, bases_description((bases[0], bases[1], bases[2])),
            bases_description(new_bases), runs_scored, outs
        )

        # Advance lineup index (batter used their PA)
        lineup_idx += 1
        pa_count   += 1

    # next_lineup_idx is already pointing to the next batter after the 3rd out
    return runs, lineup_idx % 9


def simulate_half_inning_with_bullpen(
    lineup:        list[BatterProfile],
    lineup_idx:    int,
    starter:       PitcherProfile,
    bullpen_era:   float,
    park:          ParkContext,
    rng:           np.random.Generator,
    starter_done:  bool = False,
    ghost_runner:  bool = False,
) -> tuple[int, int]:
    """
    Simulate a half-inning where the starter may be replaced mid-inning by the bullpen.
    Currently: bullpen takes over for the full inning (no mid-inning change modeled).
    The active pitcher is passed in by game_simulator.
    """
    # This wrapper exists for future mid-inning pitching change support.
    # For now, game_simulator handles pitcher switching between innings.
    return simulate_half_inning(
        lineup=lineup,
        lineup_idx=lineup_idx,
        pitcher=starter,
        park=park,
        rng=rng,
        ghost_runner=ghost_runner,
    )
