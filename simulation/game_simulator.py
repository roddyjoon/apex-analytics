"""
Apex Analytics — Full Game Simulator
Simulates one complete MLB game: 9 innings + extra innings if tied.

Key mechanics:
  - Dynamic starter removal (probabilistic, via pitcher_removal.py)
  - Bullpen takes over when starter is pulled (synthetic pitcher from BullpenProfile)
  - Extra innings: ghost runner on 2B (MLB rule since 2020)
  - Walk-off: home team ends game the moment they take the lead in the 9th+
  - Home team skips bottom 9th if they're ahead after top 9th
"""

import logging
from copy import deepcopy
from dataclasses import replace
from typing import Optional

import numpy as np

from config import (
    EXTRA_INNINGS_CAP,
    EXTRA_INNINGS_SCORING_RATE,
    HOME_FIELD_ADVANTAGE,
    FATIGUE_PER_INNING,
    MAX_FATIGUE_PENALTY,
)
from simulation.profiles import (
    BatterProfile, PitcherProfile, BullpenProfile, ParkContext, GameContext
)
from simulation.inning_simulator import simulate_half_inning
from simulation.pitcher_removal import should_remove_starter, reset_pitcher_fatigue

logger = logging.getLogger(__name__)


def simulate_game(context: GameContext, rng: np.random.Generator) -> dict:
    """
    Simulate one full MLB game.

    Parameters
    ----------
    context : GameContext — fully assembled game object with all profiles.
    rng     : numpy random Generator (caller provides seeded RNG).

    Returns
    -------
    dict with keys:
      home_score    : int
      away_score    : int
      home_win      : bool
      total_runs    : int
      innings_played: int
      inning_scores : list of (away_runs, home_runs) per inning
      went_extra    : bool
    """
    if not _validate_context(context):
        return _default_result()

    # Deep copy profiles so we can mutate fatigue/pitch counts without affecting originals
    home_starter = _copy_pitcher(context.home_starter) or _league_avg_starter(is_home=True)
    away_starter = _copy_pitcher(context.away_starter) or _league_avg_starter(is_home=False)
    home_lineup  = context.home_lineup
    away_lineup  = context.away_lineup
    park         = context.park

    # Home field advantage encoded in starter ERA adjustment
    # (Small adjustment — main HFA is in Elo layer)
    if home_starter:
        home_starter.true_talent_era *= (1.0 - HOME_FIELD_ADVANTAGE * 0.5)

    # Bullpen pitchers (synthetic profiles from BullpenProfile)
    home_bullpen_pitcher = _bullpen_to_pitcher(context.home_bullpen)
    away_bullpen_pitcher = _bullpen_to_pitcher(context.away_bullpen)

    # Game state
    home_score    = 0
    away_score    = 0
    inning_scores = []

    # Lineup position trackers (where in the batting order each team is)
    away_lineup_idx = 0
    home_lineup_idx = 0

    # Active pitchers
    home_pitcher = home_starter
    away_pitcher = away_starter
    home_starter_done = False
    away_starter_done = False

    # Runs allowed by current starter (for removal decision)
    home_starter_runs = 0
    away_starter_runs = 0

    # ── Innings 1–9 ───────────────────────────────────────────────────────────
    for inning in range(1, 10):

        # ── Top of inning: away team bats vs home pitcher ─────────────────────
        away_runs, away_lineup_idx = simulate_half_inning(
            lineup=away_lineup,
            lineup_idx=away_lineup_idx,
            pitcher=home_pitcher,
            park=park,
            rng=rng,
        )
        away_score += away_runs
        if not home_starter_done:
            home_starter_runs += away_runs

        # Check starter removal after top half
        if not home_starter_done and home_starter:
            if should_remove_starter(home_starter, inning, home_starter_runs, rng):
                home_pitcher = home_bullpen_pitcher
                home_starter_done = True
                logger.debug("Home starter removed after top of inning %d.", inning)

        # ── Bottom of inning: home team bats vs away pitcher ──────────────────

        # Walk-off check: if it's inning 9+ and home is already losing,
        # they need to bat. If they're already ahead, they don't bat.
        if inning >= 9 and home_score > away_score:
            # Home team wins — no need to bat in the bottom half
            inning_scores.append((away_runs, 0))
            break

        home_runs, home_lineup_idx = simulate_half_inning(
            lineup=home_lineup,
            lineup_idx=home_lineup_idx,
            pitcher=away_pitcher,
            park=park,
            rng=rng,
        )
        home_score += home_runs
        if not away_starter_done:
            away_starter_runs += home_runs

        inning_scores.append((away_runs, home_runs))

        # Walk-off check: home takes lead mid-inning (handled in bottom half)
        if inning >= 9 and home_score > away_score:
            break  # Walk-off win

        # Check away starter removal after bottom half
        if not away_starter_done and away_starter:
            if should_remove_starter(away_starter, inning, away_starter_runs, rng):
                away_pitcher = away_bullpen_pitcher
                away_starter_done = True
                logger.debug("Away starter removed after bottom of inning %d.", inning)

    # ── Extra innings ─────────────────────────────────────────────────────────
    went_extra   = False
    extra_inning = 9

    while home_score == away_score and extra_inning < EXTRA_INNINGS_CAP:
        extra_inning += 1
        went_extra    = True

        # Both teams use bullpen in extras
        home_bp = home_bullpen_pitcher
        away_bp = away_bullpen_pitcher

        # Slightly fatigue bullpen each extra inning
        home_bp.fatigue_index = min(MAX_FATIGUE_PENALTY,
                                    home_bp.fatigue_index + FATIGUE_PER_INNING * 2)
        away_bp.fatigue_index = min(MAX_FATIGUE_PENALTY,
                                    away_bp.fatigue_index + FATIGUE_PER_INNING * 2)

        # Top: away bats with ghost runner
        away_runs, away_lineup_idx = simulate_half_inning(
            lineup=away_lineup,
            lineup_idx=away_lineup_idx,
            pitcher=home_bp,
            park=park,
            rng=rng,
            ghost_runner=True,
        )
        away_score += away_runs

        # Bottom: home bats with ghost runner
        home_runs, home_lineup_idx = simulate_half_inning(
            lineup=home_lineup,
            lineup_idx=home_lineup_idx,
            pitcher=away_bp,
            park=park,
            rng=rng,
            ghost_runner=True,
        )
        home_score += home_runs

        inning_scores.append((away_runs, home_runs))

        # Walk-off
        if home_score != away_score:
            break

    # ── Force result if we hit extra innings cap ───────────────────────────────
    if home_score == away_score:
        # Coin flip weighted by bullpen quality
        home_bp_era = getattr(context.home_bullpen, "xfip", 4.20)
        away_bp_era = getattr(context.away_bullpen, "xfip", 4.20)
        home_edge   = away_bp_era / (home_bp_era + away_bp_era)
        if rng.random() < home_edge:
            home_score += 1
        else:
            away_score += 1

    innings_played = len(inning_scores)
    home_win       = home_score > away_score

    logger.debug(
        "Game result: %s %d – %s %d | %d innings | extra=%s",
        context.away_team_abbr, away_score,
        context.home_team_abbr, home_score,
        innings_played, went_extra
    )

    return {
        "home_score":     home_score,
        "away_score":     away_score,
        "home_win":       home_win,
        "total_runs":     home_score + away_score,
        "innings_played": innings_played,
        "inning_scores":  inning_scores,
        "went_extra":     went_extra,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_pitcher(pitcher: Optional[PitcherProfile]) -> Optional[PitcherProfile]:
    """Create a shallow copy of a pitcher profile so fatigue mutations are isolated."""
    if pitcher is None:
        return None
    return PitcherProfile(
        player_id=pitcher.player_id,
        player_name=pitcher.player_name,
        throws=pitcher.throws,
        is_starter=pitcher.is_starter,
        is_opener=pitcher.is_opener,
        true_talent_era=pitcher.true_talent_era,
        xera=pitcher.xera,
        fip=pitcher.fip,
        siera=pitcher.siera,
        k_pct=pitcher.k_pct,
        bb_pct=pitcher.bb_pct,
        gb_pct=pitcher.gb_pct,
        fb_pct=pitcher.fb_pct,
        ld_pct=pitcher.ld_pct,
        swstr_pct=pitcher.swstr_pct,
        csw_pct=pitcher.csw_pct,
        barrel_pct_allowed=pitcher.barrel_pct_allowed,
        xba_allowed=pitcher.xba_allowed,
        home_era_adj=pitcher.home_era_adj,
        away_era_adj=pitcher.away_era_adj,
        exp_decay_xera=pitcher.exp_decay_xera,
        exp_decay_k_pct=pitcher.exp_decay_k_pct,
        exp_decay_bb_pct=pitcher.exp_decay_bb_pct,
        arsenal=pitcher.arsenal,
        fatigue_index=0.0,        # Reset for each simulation
        days_rest=pitcher.days_rest,
        pitch_count_est=0,        # Reset for each simulation
        stuff_plus_proxy=pitcher.stuff_plus_proxy,
        bf=pitcher.bf,
        confidence=pitcher.confidence,
        is_on_il=pitcher.is_on_il,
        is_home=pitcher.is_home,
    )


def _bullpen_to_pitcher(bullpen: Optional[BullpenProfile]) -> PitcherProfile:
    """
    Build a synthetic PitcherProfile from a BullpenProfile.
    Used when the starter is removed — the 'bullpen' is modeled as one aggregate arm.
    """
    if bullpen is None:
        return _default_bullpen_pitcher()

    # Use high-leverage xFIP as the bullpen ERA (better relievers face key spots)
    era = bullpen.high_lev_xfip

    return PitcherProfile(
        player_id=-(bullpen.team_id),   # Negative ID = synthetic bullpen
        player_name=f"{bullpen.team_abbr} Bullpen",
        throws="R",                      # Mixed; doesn't affect model significantly
        is_starter=False,
        true_talent_era=era,
        xera=era,
        fip=era,
        siera=era,
        k_pct=0.245,                     # Relievers slightly higher K%
        bb_pct=0.088,
        gb_pct=0.430,
        fb_pct=0.350,
        ld_pct=0.220,
        swstr_pct=0.125,                 # Relievers throw harder, more whiffs
        csw_pct=0.290,
        barrel_pct_allowed=0.075,
        xba_allowed=0.240,
        home_era_adj=1.0,
        away_era_adj=1.0,
        fatigue_index=0.05 if bullpen.fatigue_flag else 0.0,
        days_rest=1,
        stuff_plus_proxy=105.0,
        bf=500,
        confidence="HIGH",
        is_home=False,
    )


def _default_bullpen_pitcher() -> PitcherProfile:
    return _bullpen_to_pitcher(BullpenProfile(team_id=0, team_abbr="MLB"))


def _league_avg_starter(is_home: bool = True) -> PitcherProfile:
    """
    Return a league-average starter profile used as a fallback when no pitcher
    data is available (e.g., TBD starter, no Statcast data yet).
    ERA set to LEAGUE_AVG_ERA; all other stats at 2024 MLB averages.
    """
    from config import (
        LEAGUE_AVG_ERA, LEAGUE_AVG_K_RATE, LEAGUE_AVG_BB_RATE,
        LEAGUE_AVG_SWSTR_PCT, LEAGUE_AVG_CSW_PCT, LEAGUE_AVG_BARREL_PCT,
    )
    era = LEAGUE_AVG_ERA
    return PitcherProfile(
        player_id          = 0,
        player_name        = "TBD",
        throws             = "R",
        is_starter         = True,
        is_opener          = False,
        true_talent_era    = era,
        xera               = era,
        fip                = era,
        siera              = era,
        k_pct              = LEAGUE_AVG_K_RATE,
        bb_pct             = LEAGUE_AVG_BB_RATE,
        gb_pct             = 0.44,
        fb_pct             = 0.36,
        ld_pct             = 0.20,
        swstr_pct          = LEAGUE_AVG_SWSTR_PCT,
        csw_pct            = LEAGUE_AVG_CSW_PCT,
        barrel_pct_allowed = LEAGUE_AVG_BARREL_PCT,
        xba_allowed        = 0.248,
        home_era_adj       = 1.0,
        away_era_adj       = 1.0,
        exp_decay_xera     = era,
        exp_decay_k_pct    = LEAGUE_AVG_K_RATE,
        exp_decay_bb_pct   = LEAGUE_AVG_BB_RATE,
        arsenal            = {"FF": 0.55, "SL": 0.25, "CH": 0.15, "CU": 0.05},
        fatigue_index      = 0.0,
        days_rest          = 4,
        pitch_count_est    = 0,
        stuff_plus_proxy   = 100.0,
        bf                 = 0,
        confidence         = "LOW",
        is_on_il           = False,
        is_home            = is_home,
    )


def _validate_context(context: GameContext) -> bool:
    """Validate that the game context has minimum required data."""
    if not context.home_lineup or not context.away_lineup:
        logger.error("Game %d missing lineup data.", context.game_pk)
        return False
    if context.home_starter is None and context.away_starter is None:
        logger.warning("Game %d: both starting pitchers unknown — using league avg.", context.game_pk)
    return True


def _default_result() -> dict:
    """Return a neutral result when simulation cannot run."""
    return {
        "home_score": 4,
        "away_score": 4,
        "home_win": True,
        "total_runs": 8,
        "innings_played": 9,
        "inning_scores": [],
        "went_extra": False,
    }
