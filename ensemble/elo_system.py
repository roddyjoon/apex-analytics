"""
Apex Analytics — Team Elo Rating System
Dynamic team strength ratings updated after every game.

Why Elo:
  - Captures overall team form including factors Monte Carlo misses
    (bullpen usage patterns, lineup injuries not in Statcast yet)
  - Fast signal when betting lines haven't adjusted post-series outcomes
  - Carries pre-season strength signal into April before Statcast samples grow

Configuration:
  Starting Elo:       1500 (all teams equal at season start)
  Home field bonus:   +48 pts (applied at prediction time, not stored)
  K-factor:           20 (rating volatility — higher = faster updates)
  Season regression:  33% toward 1500 each new season
  Margin of victory:  1 + (0.05 × min(run_diff, 8)) multiplier on K
"""

import logging
from datetime import date, datetime
from typing import Optional

from config import (
    ELO_STARTING,
    ELO_HOME_FIELD_BONUS,
    ELO_K_FACTOR,
    ELO_SEASON_REGRESSION,
)
from data.cache.db import upsert_elo, get_elo

logger = logging.getLogger(__name__)

# All 30 MLB team IDs (MLB Stats API)
MLB_TEAM_IDS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


# ---------------------------------------------------------------------------
# Core Elo functions
# ---------------------------------------------------------------------------

def win_probability(home_elo: float, away_elo: float) -> float:
    """
    Compute home team win probability from Elo ratings.
    Home field bonus (+48) is applied here, not stored in the rating.

    Returns probability between 0.0 and 1.0.
    """
    adjusted_home = home_elo + ELO_HOME_FIELD_BONUS
    return 1.0 / (1.0 + 10 ** ((away_elo - adjusted_home) / 400.0))


def update_elo_ratings(
    winner_elo:       float,
    loser_elo:        float,
    home_team_won:    bool,
    run_differential: int,
) -> tuple[float, float]:
    """
    Update Elo ratings after a completed game.

    Parameters
    ----------
    winner_elo       : Elo of the winning team.
    loser_elo        : Elo of the losing team.
    home_team_won    : True if the home team won (for expected value calculation).
    run_differential : Absolute run difference (used for margin-of-victory factor).

    Returns
    -------
    (new_winner_elo, new_loser_elo)
    """
    # Margin-of-victory multiplier (capped at 8 runs to prevent blowout inflation)
    mov_factor = 1.0 + (0.05 * min(abs(run_differential), 8))
    k_adjusted = ELO_K_FACTOR * mov_factor

    # Expected win probability (from winner's perspective, with HFA)
    if home_team_won:
        # Winner was home team
        expected_winner = win_probability(winner_elo, loser_elo)
    else:
        # Winner was away team (no HFA — use raw Elo diff)
        expected_winner = 1.0 / (1.0 + 10 ** ((loser_elo - winner_elo) / 400.0))

    # Update ratings
    new_winner_elo = winner_elo + k_adjusted * (1.0 - expected_winner)
    new_loser_elo  = loser_elo  + k_adjusted * (0.0 - (1.0 - expected_winner))

    logger.debug(
        "Elo update: winner %.1f → %.1f | loser %.1f → %.1f | "
        "MOV_factor=%.2f | expected=%.3f",
        winner_elo, new_winner_elo, loser_elo, new_loser_elo,
        mov_factor, expected_winner
    )
    return new_winner_elo, new_loser_elo


def regress_to_mean(elo: float, regression_pct: float = ELO_SEASON_REGRESSION) -> float:
    """
    Regress an Elo rating toward 1500 at the start of a new season.
    33% regression → a team at 1600 becomes 1567 (1600 - 0.33×100).
    """
    return elo + regression_pct * (ELO_STARTING - elo)


# ---------------------------------------------------------------------------
# DB-backed operations
# ---------------------------------------------------------------------------

def get_team_elo(team_id: int, season: int) -> float:
    """
    Get current Elo rating for a team. Returns ELO_STARTING if not found.
    """
    elo = get_elo(team_id, season)
    if elo is None:
        logger.debug("No Elo found for team %d season %d — returning default %d.",
                     team_id, season, ELO_STARTING)
        return float(ELO_STARTING)
    return elo


def process_game_result(
    home_team_id:  int,
    away_team_id:  int,
    home_score:    int,
    away_score:    int,
    season:        int,
    home_team_abbr: str = "",
    away_team_abbr: str = "",
) -> dict:
    """
    Process a completed game and update both teams' Elo ratings.

    Parameters
    ----------
    home_team_id  : MLB Stats API team ID for home team.
    away_team_id  : MLB Stats API team ID for away team.
    home_score    : Final home team score.
    away_score    : Final away team score.
    season        : Season year.

    Returns
    -------
    dict with old/new ratings for both teams and the updated win probability.
    """
    if home_score == away_score:
        logger.warning("Tied game passed to Elo update — skipping (game not final?).")
        return {}

    home_elo = get_team_elo(home_team_id, season)
    away_elo = get_team_elo(away_team_id, season)

    home_won        = home_score > away_score
    run_differential = abs(home_score - away_score)

    if home_won:
        new_home_elo, new_away_elo = update_elo_ratings(
            winner_elo=home_elo, loser_elo=away_elo,
            home_team_won=True, run_differential=run_differential
        )
    else:
        new_away_elo, new_home_elo = update_elo_ratings(
            winner_elo=away_elo, loser_elo=home_elo,
            home_team_won=False, run_differential=run_differential
        )

    # Persist to DB
    upsert_elo(home_team_id, home_team_abbr or str(home_team_id),
               season, new_home_elo, games_played=0)
    upsert_elo(away_team_id, away_team_abbr or str(away_team_id),
               season, new_away_elo, games_played=0)

    result = {
        "home_team_id":    home_team_id,
        "away_team_id":    away_team_id,
        "home_elo_before": round(home_elo, 1),
        "away_elo_before": round(away_elo, 1),
        "home_elo_after":  round(new_home_elo, 1),
        "away_elo_after":  round(new_away_elo, 1),
        "home_won":        home_won,
        "run_differential": run_differential,
        "elo_win_prob_before": round(win_probability(home_elo, away_elo), 4),
        "elo_win_prob_after":  round(win_probability(new_home_elo, new_away_elo), 4),
    }
    logger.info(
        "Elo update: %s %.1f→%.1f | %s %.1f→%.1f | result: %s won by %d",
        home_team_abbr or home_team_id, home_elo, new_home_elo,
        away_team_abbr or away_team_id, away_elo, new_away_elo,
        ("home" if home_won else "away"), run_differential
    )
    return result


def initialize_season_elos(season: int, team_ids: Optional[list[int]] = None) -> None:
    """
    Initialize (or regress) Elo ratings at the start of a new season.
    - Teams with existing prior-season Elo: regress 33% toward 1500.
    - Teams with no prior data: set to 1500.
    """
    if team_ids is None:
        team_ids = list(MLB_TEAM_IDS.keys())

    prior_season = season - 1
    for team_id in team_ids:
        abbr = MLB_TEAM_IDS.get(team_id, str(team_id))
        prior_elo = get_elo(team_id, prior_season)
        if prior_elo is not None:
            new_elo = regress_to_mean(prior_elo)
            logger.info("Season regression: %s %.1f → %.1f", abbr, prior_elo, new_elo)
        else:
            new_elo = float(ELO_STARTING)
            logger.info("New Elo: %s initialized at %.1f", abbr, new_elo)
        upsert_elo(team_id, abbr, season, new_elo, games_played=0)


def get_all_elos(season: int) -> dict[int, float]:
    """Return {team_id: elo} for all teams in a season."""
    result = {}
    for team_id in MLB_TEAM_IDS:
        result[team_id] = get_team_elo(team_id, season)
    return result


def elo_to_win_pct(elo: float) -> float:
    """
    Convert Elo rating to approximate win percentage (vs. average 1500 opponent).
    Useful for display/debugging.
    """
    return win_probability(elo, ELO_STARTING)


def batch_update_from_results(results: list[dict], season: int) -> list[dict]:
    """
    Process a list of game result dicts (from mlb_results.py) and update all Elo ratings.

    Each result dict should have:
      game_pk, home_team_id, home_team_abbr, away_team_id, away_team_abbr,
      home_score, away_score, home_win

    Returns list of Elo update records.
    """
    updates = []
    for result in results:
        if result.get("home_score") is None or result.get("away_score") is None:
            continue
        if result["home_score"] == result["away_score"]:
            continue  # Skip ties (shouldn't happen in MLB)

        update = process_game_result(
            home_team_id=result["home_team_id"],
            away_team_id=result["away_team_id"],
            home_score=result["home_score"],
            away_score=result["away_score"],
            season=season,
            home_team_abbr=result.get("home_team_abbr", ""),
            away_team_abbr=result.get("away_team_abbr", ""),
        )
        if update:
            updates.append(update)

    logger.info("Batch Elo update: processed %d games.", len(updates))
    return updates
