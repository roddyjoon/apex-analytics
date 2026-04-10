"""
Apex Analytics — MLB Roster Ingestor
Fetches active 26-man roster + IL status per team from MLB Stats API.
Used for:
  - Lineup fallback (sort by xwOBA when no historical batting order)
  - IL tracking (flag injured players in report)
  - September 28-man roster detection
"""

import logging
from datetime import date
from typing import Optional

import requests

from data.cache.db import upsert_player
from data.cache.file_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
ROSTER_CACHE_TTL_HOURS = 4.0   # Rosters don't change minute-to-minute


def fetch_roster(team_id: int, game_date: Optional[date] = None) -> dict:
    """
    Fetch active roster + IL for a team.

    Returns:
        {
          "active": [
            {"player_id": int, "full_name": str, "position": str,
             "bats": str, "throws": str, "status": "active"},
            ...
          ],
          "il": [
            {"player_id": int, "full_name": str, "position": str,
             "status": "10-day IL" | "15-day IL" | "60-day IL"},
            ...
          ],
          "roster_size": int,
          "is_september": bool,
        }
    """
    if game_date is None:
        game_date = date.today()

    cache_key = f"roster_{team_id}_{game_date.isoformat()}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    is_september = game_date.month >= 9
    roster_type = "active"  # Always use "active" roster type; filter IL separately

    # Active 26-man (or 28-man September) roster
    url = f"{MLB_API_BASE}/teams/{team_id}/roster/{roster_type}"
    params = {"date": game_date.isoformat(),
              "fields": "roster,person,id,fullName,primaryPosition,code,abbreviation,batSide,pitchHand"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Roster fetch failed for team %d: %s", team_id, exc)
        return _empty_roster_response(is_september)

    active_players = []
    for entry in data.get("roster", []):
        person = entry.get("person", {})
        pos    = entry.get("position", {})
        player = {
            "player_id": person.get("id"),
            "full_name": person.get("fullName", ""),
            "position":  pos.get("abbreviation", ""),
            "bats":      person.get("batSide", {}).get("code", "R"),
            "throws":    person.get("pitchHand", {}).get("code", "R"),
            "status":    "active",
            "on_il":     False,
        }
        active_players.append(player)
        # Persist to players table
        upsert_player({
            "player_id": player["player_id"],
            "full_name": player["full_name"],
            "position":  player["position"],
            "bats":      player["bats"],
            "throws":    player["throws"],
            "team_id":   team_id,
            "active":    True,
            "on_il":     False,
        })

    # Fetch IL separately
    il_players = _fetch_il(team_id, game_date)

    result = {
        "active": active_players,
        "il": il_players,
        "roster_size": len(active_players),
        "is_september": is_september,
    }

    set_cache(cache_key, result, ttl_hours=ROSTER_CACHE_TTL_HOURS)
    return result


def _fetch_il(team_id: int, game_date: date) -> list[dict]:
    """Fetch injured list entries for a team."""
    url = f"{MLB_API_BASE}/teams/{team_id}/roster/injuries"
    params = {"date": game_date.isoformat()}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("IL fetch failed for team %d: %s", team_id, exc)
        return []

    il_players = []
    for entry in data.get("roster", []):
        person = entry.get("person", {})
        pos    = entry.get("position", {})
        status = entry.get("status", {}).get("description", "IL")
        player = {
            "player_id": person.get("id"),
            "full_name": person.get("fullName", ""),
            "position":  pos.get("abbreviation", ""),
            "status":    status,
            "on_il":     True,
        }
        il_players.append(player)
        # Update IL flag in players table
        upsert_player({
            "player_id": player["player_id"],
            "full_name": player["full_name"],
            "position":  player["position"],
            "team_id":   team_id,
            "active":    True,
            "on_il":     True,
        })
    return il_players


def get_all_team_rosters(team_ids: list[int], game_date: Optional[date] = None) -> dict[int, dict]:
    """Fetch rosters for multiple teams. Returns {team_id: roster_dict}."""
    if game_date is None:
        game_date = date.today()
    return {tid: fetch_roster(tid, game_date) for tid in team_ids}


def get_active_player_ids(team_id: int, game_date: Optional[date] = None) -> list[int]:
    """Return list of active player IDs for a team (excludes IL)."""
    roster = fetch_roster(team_id, game_date)
    return [p["player_id"] for p in roster["active"] if p["player_id"] is not None]


def is_player_on_il(player_id: int, team_id: int, game_date: Optional[date] = None) -> bool:
    """Quick check if a specific player is on the IL."""
    roster = fetch_roster(team_id, game_date)
    il_ids = {p["player_id"] for p in roster["il"]}
    return player_id in il_ids


def _empty_roster_response(is_september: bool = False) -> dict:
    return {"active": [], "il": [], "roster_size": 0, "is_september": is_september}
