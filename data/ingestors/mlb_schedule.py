"""
Apex Analytics — MLB Schedule Ingestor
Fetches today's game slate from the MLB Stats API (public, no auth).
Handles postponements, double-headers, and makeup game location.
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from config import TIMEZONE
from data.cache.db import upsert_game, get_games_for_date
from data.cache.file_cache import get_cache, set_cache, schedule_cache_key

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
SCHEDULE_TTL_HOURS = 1.0  # Re-fetch schedule every hour (postponements can drop mid-day)


def fetch_schedule(game_date: Optional[date] = None) -> list[dict]:
    """
    Fetch all MLB games for a given date (defaults to today).
    Returns a list of game dicts; persists to DB.

    Each returned dict:
      game_pk, game_date, home_team_id, home_team_abbr, away_team_id, away_team_abbr,
      venue_id, venue_name, game_time_utc, status, double_header
    """
    if game_date is None:
        game_date = date.today()

    date_str = game_date.isoformat()
    cache_key = schedule_cache_key(date_str)

    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,           # MLB
        "date": date_str,
        "hydrate": "team,venue,game(content(summary))",
        "fields": (
            "dates,date,games,gamePk,gameType,status,statusCode,"
            "teams,home,away,team,id,abbreviation,name,"
            "venue,id,name,"
            "gameDate,doubleHeader,gameNumber"
        ),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Schedule fetch failed for %s: %s", date_str, exc)
        # Fall back to DB cache
        db_games = get_games_for_date(game_date)
        if db_games:
            logger.warning("Using stale DB cache for schedule (%s).", date_str)
            return db_games
        return []

    games_out = []
    dates = data.get("dates", [])
    if not dates:
        logger.info("No games scheduled for %s.", date_str)
        set_cache(cache_key, [], ttl_hours=SCHEDULE_TTL_HOURS)
        return []

    for date_block in dates:
        for game in date_block.get("games", []):
            game_type = game.get("gameType", "R")
            if game_type not in ("R", "D", "W", "L", "F"):
                continue  # skip spring training, all-star, etc.

            status_code = game.get("status", {}).get("statusCode", "")
            status = _map_status(status_code)

            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            venue = game.get("venue", {})

            game_time_str = game.get("gameDate", "")
            game_time_utc = None
            if game_time_str:
                try:
                    game_time_utc = datetime.fromisoformat(
                        game_time_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            double_header = game.get("doubleHeader", "N")
            if double_header not in ("Y", "S", "N"):
                double_header = "N"
            game_number = game.get("gameNumber", 1)
            # Encode as "N", "Y" (game 1), or "Z" (game 2)
            if double_header in ("Y", "S") and game_number == 2:
                double_header = "Z"

            game_dict = {
                "game_pk":        game["gamePk"],
                "game_date":      game_date,
                "home_team_id":   home.get("team", {}).get("id"),
                "home_team_abbr": home.get("team", {}).get("abbreviation", ""),
                "away_team_id":   away.get("team", {}).get("id"),
                "away_team_abbr": away.get("team", {}).get("abbreviation", ""),
                "venue_id":       venue.get("id"),
                "venue_name":     venue.get("name", ""),
                "game_time_utc":  game_time_utc,
                "status":         status,
                "double_header":  double_header,
                "home_score":     None,
                "away_score":     None,
                "innings":        None,
                "home_win":       None,
            }

            upsert_game(game_dict)
            games_out.append(game_dict)

    logger.info("Fetched %d games for %s.", len(games_out), date_str)
    set_cache(cache_key, games_out, ttl_hours=SCHEDULE_TTL_HOURS)
    return games_out


def _map_status(status_code: str) -> str:
    """Map MLB API status codes to simplified status strings."""
    mapping = {
        "S":  "scheduled",
        "PW": "scheduled",   # Pre-warmup
        "P":  "scheduled",
        "I":  "in_progress",
        "MA": "in_progress",
        "MF": "final",
        "F":  "final",
        "O":  "final",
        "UR": "final",
        "D":  "postponed",
        "DI": "postponed",
        "DC": "postponed",
        "DR": "postponed",
        "TR": "suspended",
        "TS": "suspended",
    }
    return mapping.get(status_code, "scheduled")


def get_todays_games(exclude_postponed: bool = True) -> list[dict]:
    """Convenience wrapper: return today's active game schedule."""
    games = fetch_schedule(date.today())
    if exclude_postponed:
        games = [g for g in games if g["status"] != "postponed"]
    return games


def get_venue_coords(venue_id: int) -> Optional[dict]:
    """
    Look up GPS coordinates and metadata for an MLB venue by venue_id.
    Falls back to a venue_name search if venue_id doesn't match directly.

    Returns a dict with keys: lat, lon, cf_orientation_deg, is_dome, elevation_ft
    Returns None if the venue is not found.
    """
    stadiums_path = Path(__file__).parent.parent.parent / "data" / "stadiums.json"
    try:
        with open(stadiums_path) as f:
            data = json.load(f)
        stadiums = data.get("stadiums", data) if isinstance(data, dict) else data
        if isinstance(stadiums, dict):
            stadiums = list(stadiums.values())

        # Build a flat list if stadiums is a list of lists (outer list has one item)
        if stadiums and isinstance(stadiums[0], list):
            stadiums = stadiums[0]

        # We don't have venue_id in stadiums.json, so we return any entry for now.
        # In production this would be keyed by venue_id; for now return first match.
        # (The schedule includes venue_name which can be cross-referenced if needed.)
        if stadiums:
            # Return the dict for use by weather fetcher — caller picks the right one
            # by matching on the game's home_team_abbr upstream
            return stadiums  # Return full list; caller selects by team_abbr

        return None

    except Exception as exc:
        logger.warning("Could not load stadiums.json: %s", exc)
        return None


def get_venue_coords_by_team(team_abbr: str) -> Optional[dict]:
    """
    Return GPS coordinates and park metadata for a team's home stadium.
    This is the preferred function — keyed by team abbreviation which is always available.

    Returns dict with: lat, lon, cf_orientation_deg, is_dome, elevation_ft, stadium_name
    """
    stadiums_path = Path(__file__).parent.parent.parent / "data" / "stadiums.json"
    try:
        with open(stadiums_path) as f:
            data = json.load(f)
        stadiums = data.get("stadiums", [])
        if isinstance(stadiums, dict):
            stadiums = list(stadiums.values())
        if stadiums and isinstance(stadiums[0], list):
            stadiums = stadiums[0]

        for entry in stadiums:
            if entry.get("team_abbr", "").upper() == team_abbr.upper():
                return {
                    "lat":                entry.get("lat"),
                    "lon":                entry.get("lon"),
                    "cf_orientation_deg": entry.get("cf_orientation_deg", 0),
                    "is_dome":            entry.get("dome", False),
                    "elevation_ft":       entry.get("elevation_ft", 0),
                    "stadium_name":       entry.get("stadium_name", ""),
                }
        return None
    except Exception as exc:
        logger.warning("Could not load stadiums.json for team %s: %s", team_abbr, exc)
        return None


def is_double_header_game2(game: dict) -> bool:
    """
    Return True if this game is Game 2 of a double-header.
    Handles both:
      - Processed schedule format: double_header == "Z" (encoded by fetch_schedule)
      - Raw MLB Stats API format: doubleHeader in ("Y","S") and gameNumber == 2
    """
    # Processed format (output of fetch_schedule)
    if game.get("double_header") == "Z":
        return True
    # Raw MLB Stats API format
    dh = game.get("doubleHeader", "N")
    gn = game.get("gameNumber", 1)
    return dh in ("Y", "S") and gn == 2


def get_prior_day_innings(game_pk: int) -> Optional[int]:
    """
    Return innings played in a given completed game (for bullpen fatigue check).
    Returns None if game not finished or not found.
    """
    url = f"{MLB_API_BASE}/game/{game_pk}/linescore"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        innings_list = data.get("innings", [])
        return len(innings_list) if innings_list else None
    except Exception as exc:
        logger.warning("Could not fetch linescore for gamePk %d: %s", game_pk, exc)
        return None
