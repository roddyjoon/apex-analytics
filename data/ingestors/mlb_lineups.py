"""
Apex Analytics — MLB Lineup Ingestor
Polls MLB Stats API for confirmed batting orders.
Returns empty gracefully when lineups not yet posted (morning report scenario).
Detects openers / bullpen games.
"""

import logging
from datetime import date, datetime
from typing import Optional

import requests

from data.cache.db import save_lineups, get_lineups
from data.cache.file_cache import get_cache, set_cache, lineup_cache_key

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
LINEUP_CACHE_TTL_HOURS = 0.5  # Poll every 30 min on game day


def fetch_confirmed_lineup(game_pk: int, report_type: str = "morning") -> dict:
    """
    Fetch confirmed batting lineup for a game from MLB Stats API boxscore endpoint.

    Returns:
        {
          "home": {
            "team_id": int,
            "team_abbr": str,
            "lineup": [
              {"batting_order": 1, "player_id": int, "player_name": str,
               "position": str, "is_confirmed": True, "source": "api"},
              ...
            ],
            "sp_player_id": int | None,
            "sp_player_name": str | None,
            "sp_position_type": str,   # "SP" or "RP" (opener detection)
            "is_confirmed": bool,
          },
          "away": { ... same structure ... },
        }
    Returns empty lineups ({home: {lineup: []}, away: {lineup: []}}) if not posted.
    """
    cache_key = lineup_cache_key(game_pk, report_type)
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("Lineup fetch failed for gamePk %d: %s", game_pk, exc)
        return _empty_lineup_response()

    teams_data = data.get("teams", {})
    result = {}

    for side in ("home", "away"):
        team_data = teams_data.get(side, {})
        team_info = team_data.get("team", {})
        team_id   = team_info.get("id")
        team_abbr = team_info.get("abbreviation", "")

        # Batting order is in "battingOrder" field (list of player IDs in order)
        batting_order_ids = team_data.get("battingOrder", [])
        players_dict = team_data.get("players", {})

        lineup = []
        for position_idx, player_id in enumerate(batting_order_ids, start=1):
            player_key = f"ID{player_id}"
            player_info = players_dict.get(player_key, {})
            person = player_info.get("person", {})
            pos_info = player_info.get("position", {})
            lineup.append({
                "batting_order": position_idx,
                "player_id": player_id,
                "player_name": person.get("fullName", "Unknown"),
                "position": pos_info.get("abbreviation", ""),
                "is_confirmed": True,
                "source": "api",
            })

        # Detect starting pitcher
        sp_player_id = None
        sp_player_name = None
        sp_position_type = "SP"
        pitchers = team_data.get("pitchers", [])
        if pitchers:
            sp_id = pitchers[0]  # First pitcher in list = starter
            sp_key = f"ID{sp_id}"
            sp_info = players_dict.get(sp_key, {})
            person = sp_info.get("person", {})
            sp_player_id = sp_id
            sp_player_name = person.get("fullName", "Unknown")
            # Opener detection: if listed position type is "RP" they're opening
            pos_type = sp_info.get("position", {}).get("type", "")
            if pos_type == "Pitcher":
                # Check abbreviation for RP vs SP
                pos_abbr = sp_info.get("position", {}).get("abbreviation", "SP")
                sp_position_type = "RP" if pos_abbr == "RP" else "SP"
            else:
                sp_position_type = "SP"

        is_confirmed = len(lineup) > 0

        side_result = {
            "team_id": team_id,
            "team_abbr": team_abbr,
            "lineup": lineup,
            "sp_player_id": sp_player_id,
            "sp_player_name": sp_player_name,
            "sp_position_type": sp_position_type,
            "is_confirmed": is_confirmed,
        }
        result[side] = side_result

        # Persist confirmed lineups to DB
        if is_confirmed and team_id:
            save_lineups(
                game_pk=game_pk,
                team_id=team_id,
                team_abbr=team_abbr,
                lineup=lineup,
                report_type=report_type,
            )

    # Cache briefly — re-poll frequently to catch late lineup changes
    cache_ttl = 0.25 if not result.get("home", {}).get("is_confirmed") else LINEUP_CACHE_TTL_HOURS
    set_cache(cache_key, result, ttl_hours=cache_ttl)
    return result


def fetch_probable_pitchers(game_pk: int) -> dict:
    """
    Fetch probable pitchers for a scheduled game (available before confirmed lineup).
    Returns {"home": {"player_id": int, "player_name": str}, "away": {...}}

    Strategy:
      1. Primary: schedule endpoint with hydrate=probablePitcher (works pre-game)
      2. Fallback: game feed/live endpoint (works once game feed is initialized)
    """
    _empty = {"home": {"player_id": None, "player_name": "TBD"},
              "away": {"player_id": None, "player_name": "TBD"}}

    # ── Strategy 1: schedule hydration (works for scheduled/upcoming games) ──
    try:
        url = f"{MLB_API_BASE}/schedule"
        params = {"sportId": 1, "gamePk": game_pk, "hydrate": "probablePitcher"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        dates = data.get("dates", [])
        if dates:
            game = dates[0].get("games", [{}])[0]
            teams = game.get("teams", {})
            home_p = teams.get("home", {}).get("probablePitcher", {})
            away_p = teams.get("away", {}).get("probablePitcher", {})
            if home_p or away_p:
                return {
                    "home": {"player_id": home_p.get("id"), "player_name": home_p.get("fullName", "TBD")},
                    "away": {"player_id": away_p.get("id"), "player_name": away_p.get("fullName", "TBD")},
                }
    except Exception:
        pass  # Fall through to strategy 2

    # ── Strategy 2: live game feed (works once game feed is initialized) ──
    try:
        url = f"{MLB_API_BASE}/game/{game_pk}/feed/live"
        params = {"fields": "gameData,probablePitchers,fullName,id"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        probable = data.get("gameData", {}).get("probablePitchers", {})
        if probable:
            return {
                "home": {"player_id": probable.get("home", {}).get("id"),
                         "player_name": probable.get("home", {}).get("fullName", "TBD")},
                "away": {"player_id": probable.get("away", {}).get("id"),
                         "player_name": probable.get("away", {}).get("fullName", "TBD")},
            }
    except Exception as exc:
        logger.warning("Could not fetch probable pitchers for gamePk %d: %s", game_pk, exc)

    return _empty


def is_opener_game(sp_position_type: str) -> bool:
    """
    Return True if the announced 'starter' is actually an opener (relief pitcher).
    Accepts both the short code ("RP") and the full MLB API position type
    strings ("Relief Pitcher", "Reliever") as equivalent indicators.
    """
    return sp_position_type in ("RP", "Relief Pitcher", "Reliever")


def _empty_lineup_response() -> dict:
    return {
        "home": {"team_id": None, "team_abbr": "", "lineup": [],
                 "sp_player_id": None, "sp_player_name": None,
                 "sp_position_type": "SP", "is_confirmed": False},
        "away": {"team_id": None, "team_abbr": "", "lineup": [],
                 "sp_player_id": None, "sp_player_name": None,
                 "sp_position_type": "SP", "is_confirmed": False},
    }
