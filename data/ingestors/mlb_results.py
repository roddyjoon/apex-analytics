"""
Apex Analytics — MLB Game Results Ingestor
Fetches post-game linescore for completed games.
Used for:
  - Elo rating updates (post-game nightly)
  - Calibration history recording (predicted vs. actual)
  - Bullpen fatigue detection (innings played yesterday)
"""

import logging
from datetime import date, timedelta
from typing import Optional

import requests

from data.cache.db import upsert_game, get_games_for_date
from data.cache.file_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
RESULTS_CACHE_TTL_HOURS = 12.0  # Final scores don't change


def fetch_game_result(game_pk: int) -> Optional[dict]:
    """
    Fetch linescore for a completed game.

    Returns:
        {
          "game_pk": int,
          "home_score": int,
          "away_score": int,
          "innings": int,        # Total innings played (9, 10, 11, ...)
          "home_win": bool,
          "status": "final",
        }
    Returns None if game not yet final.
    """
    cache_key = f"result_{game_pk}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    url = f"{MLB_API_BASE}/game/{game_pk}/linescore"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("Result fetch failed for gamePk %d: %s", game_pk, exc)
        return None

    # Check if game is truly final
    current_inning = data.get("currentInning", 0)
    inning_state = data.get("inningState", "")
    is_final = (inning_state.lower() in ("end", "final") or current_inning >= 9)

    # If game has no runs data, it may not be complete
    teams = data.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})

    home_score = home.get("runs")
    away_score = away.get("runs")
    innings_list = data.get("innings", [])

    if home_score is None or away_score is None:
        return None  # Game not yet started or data incomplete

    result = {
        "game_pk": game_pk,
        "home_score": home_score,
        "away_score": away_score,
        "innings": len(innings_list) if innings_list else current_inning,
        "home_win": home_score > away_score,
        "status": "final" if is_final else "in_progress",
    }

    if is_final:
        # Persist to DB
        upsert_game({
            "game_pk": game_pk,
            "home_score": home_score,
            "away_score": away_score,
            "innings": result["innings"],
            "home_win": result["home_win"],
            "status": "final",
        })
        set_cache(cache_key, result, ttl_hours=RESULTS_CACHE_TTL_HOURS)

    return result


def fetch_all_results_for_date(game_date: date) -> list[dict]:
    """
    Fetch results for all games on a given date.
    Returns list of result dicts for completed games only.
    """
    games = get_games_for_date(game_date)
    results = []
    for game in games:
        if game["status"] == "final":
            result = fetch_game_result(game["game_pk"])
            if result:
                results.append(result)
    return results


def fetch_yesterday_results() -> list[dict]:
    """Fetch results from yesterday — used in morning Elo update."""
    yesterday = date.today() - timedelta(days=1)
    return fetch_all_results_for_date(yesterday)


def get_prior_night_max_innings(team_id: int, game_date: date) -> Optional[int]:
    """
    Return innings played in the most recent completed game for a team.
    Used for bullpen fatigue detection.
    If the game went extra innings (>= 10), the bullpen fatigue flag triggers.
    """
    yesterday = game_date - timedelta(days=1)
    games = get_games_for_date(yesterday)

    # Find the game for this team
    team_games = [
        g for g in games
        if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id
    ]

    if not team_games:
        return None

    # If the team played yesterday, get innings
    game = team_games[0]
    if game.get("innings"):
        return game["innings"]

    # Try fetching live
    result = fetch_game_result(game["game_pk"])
    return result["innings"] if result else None


def run_daily_result_update(game_date: Optional[date] = None) -> list[dict]:
    """
    Main nightly job: fetch all game results from yesterday,
    return them for Elo update processing.
    """
    if game_date is None:
        game_date = date.today() - timedelta(days=1)

    logger.info("Fetching game results for %s...", game_date.isoformat())
    results = fetch_all_results_for_date(game_date)
    logger.info("Retrieved %d final results for %s.", len(results), game_date.isoformat())
    return results
