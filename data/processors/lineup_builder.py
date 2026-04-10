"""
Apex Analytics — Lineup Builder
Three-tier fallback logic for batting order assembly:

  Tier 1: MLB Stats API confirmed lineup (batting order 1–9)
  Tier 2: Historical modal batting order (last 5 game logs from Savant)
  Tier 3: Team roster sorted by season xwOBA desc (last resort)

Each slot is tagged with is_confirmed flag and source.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import requests

from config import LINEUP_FALLBACK_GAMES
from data.cache.db import get_lineups, save_lineups
from data.cache.file_cache import get_cache, set_cache, lineup_cache_key
from data.ingestors.mlb_lineups import fetch_confirmed_lineup, fetch_probable_pitchers
from data.ingestors.mlb_rosters import fetch_roster, get_active_player_ids
from data.ingestors.statcast_batter import fetch_batter_stats

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
HISTORICAL_ORDER_CACHE_TTL = 12.0  # Hours


def build_lineup(
    game_pk:     int,
    team_id:     int,
    team_abbr:   str,
    game_date:   Optional[date] = None,
    report_type: str = "morning",
    season:      int = 2025,
) -> dict:
    """
    Build a batting lineup for a team using three-tier fallback.

    Returns:
        {
          "lineup": [
            {"batting_order": 1, "player_id": int, "player_name": str,
             "position": str, "is_confirmed": bool, "source": str},
            ...
          ],
          "is_confirmed": bool,        # True if all 9 slots are confirmed
          "confidence_level": str,     # "CONFIRMED" / "HISTORICAL" / "ROSTER"
          "source": str,
        }
    """
    if game_date is None:
        game_date = date.today()

    # -------------------------------------------------------------------------
    # Tier 1: Confirmed lineup from MLB Stats API
    # -------------------------------------------------------------------------
    confirmed_data = fetch_confirmed_lineup(game_pk, report_type)
    side = _detect_side(confirmed_data, team_id, team_abbr)
    lineup_data = confirmed_data.get(side, {})
    confirmed_lineup = lineup_data.get("lineup", [])

    if len(confirmed_lineup) >= 7:  # At least 7 of 9 confirmed = use it
        logger.info("%s lineup confirmed from MLB API (%d slots).", team_abbr, len(confirmed_lineup))
        result = {
            "lineup": confirmed_lineup,
            "is_confirmed": True,
            "confidence_level": "CONFIRMED",
            "source": "api",
        }
        _persist_lineup(game_pk, team_id, team_abbr, confirmed_lineup, report_type)
        return result

    # -------------------------------------------------------------------------
    # Tier 2: Historical modal batting order (last N games)
    # -------------------------------------------------------------------------
    historical = _fetch_historical_batting_order(team_id, game_date, n_games=LINEUP_FALLBACK_GAMES)
    if len(historical) >= 7:
        logger.info("%s lineup from historical batting order (%d slots).", team_abbr, len(historical))
        # Mark as projected
        for slot in historical:
            slot["is_confirmed"] = False
            slot["source"] = "historical"
        result = {
            "lineup": historical,
            "is_confirmed": False,
            "confidence_level": "HISTORICAL",
            "source": "historical",
        }
        _persist_lineup(game_pk, team_id, team_abbr, historical, report_type)
        return result

    # -------------------------------------------------------------------------
    # Tier 3: Roster sorted by xwOBA (last resort)
    # -------------------------------------------------------------------------
    roster_lineup = _build_roster_lineup(team_id, game_date, season)
    logger.warning("%s lineup from roster fallback (%d slots).", team_abbr, len(roster_lineup))
    result = {
        "lineup": roster_lineup,
        "is_confirmed": False,
        "confidence_level": "ROSTER",
        "source": "roster",
    }
    _persist_lineup(game_pk, team_id, team_abbr, roster_lineup, report_type)
    return result


def build_both_lineups(
    game_pk:      int,
    home_team_id: int,
    home_abbr:    str,
    away_team_id: int,
    away_abbr:    str,
    game_date:    Optional[date] = None,
    report_type:  str = "morning",
    season:       int = 2025,
) -> dict:
    """
    Build lineups for both teams. Returns {"home": lineup_dict, "away": lineup_dict}.
    """
    if game_date is None:
        game_date = date.today()

    home = build_lineup(game_pk, home_team_id, home_abbr, game_date, report_type, season)
    away = build_lineup(game_pk, away_team_id, away_abbr, game_date, report_type, season)
    return {"home": home, "away": away}


# ---------------------------------------------------------------------------
# Tier 2: Historical batting order
# ---------------------------------------------------------------------------


def _fetch_historical_batting_order(
    team_id: int,
    game_date: date,
    n_games: int = 5,
) -> list[dict]:
    """
    Fetch the modal batting order from recent game logs.
    Uses MLB Stats API game feed for recent completed games.
    Returns list of slot dicts sorted by batting_order.
    """
    cache_key = f"hist_order_{team_id}_{game_date.isoformat()}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    # Fetch recent schedule for this team
    recent_game_pks = _get_recent_game_pks(team_id, game_date, n_games)
    if not recent_game_pks:
        return []

    # Collect batting orders from each game
    order_votes: dict[int, dict[int, int]] = {}  # {batting_slot: {player_id: count}}

    for game_pk in recent_game_pks:
        lineup = _fetch_game_lineup(game_pk, team_id)
        for slot in lineup:
            order = slot.get("batting_order")
            pid   = slot.get("player_id")
            if not order or not pid:
                continue
            if order not in order_votes:
                order_votes[order] = {}
            order_votes[order][pid] = order_votes[order].get(pid, 0) + 1

    if not order_votes:
        return []

    # Select modal player for each batting slot
    modal_lineup = []
    for slot in range(1, 10):
        votes = order_votes.get(slot, {})
        if not votes:
            continue
        # Most frequent player in this slot
        player_id = max(votes, key=votes.get)
        modal_lineup.append({
            "batting_order": slot,
            "player_id": player_id,
            "player_name": f"Player {player_id}",  # Name resolved in profile_builder
            "position": "",
            "is_confirmed": False,
            "source": "historical",
        })

    set_cache(cache_key, modal_lineup, ttl_hours=HISTORICAL_ORDER_CACHE_TTL)
    return modal_lineup


def _get_recent_game_pks(team_id: int, game_date: date, n: int) -> list[int]:
    """Return list of game PKs for a team's last N completed games."""
    from datetime import timedelta
    game_pks = []
    check_date = game_date - timedelta(days=1)

    while len(game_pks) < n and check_date >= game_date - timedelta(days=30):
        date_str = check_date.isoformat()
        url = f"{MLB_API_BASE}/schedule"
        params = {
            "sportId": 1,
            "date": date_str,
            "teamId": team_id,
            "fields": "dates,games,gamePk,status,statusCode",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for date_block in data.get("dates", []):
                for game in date_block.get("games", []):
                    status = game.get("status", {}).get("statusCode", "")
                    if status in ("F", "O", "MF", "UR"):  # Final
                        game_pks.append(game["gamePk"])
        except Exception:
            pass
        check_date -= timedelta(days=1)

    return game_pks[:n]


def _fetch_game_lineup(game_pk: int, team_id: int) -> list[dict]:
    """Fetch confirmed lineup from a completed game's boxscore."""
    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    teams_data = data.get("teams", {})
    for side in ("home", "away"):
        team_info = teams_data.get(side, {}).get("team", {})
        if team_info.get("id") == team_id:
            batting_ids = teams_data[side].get("battingOrder", [])
            players     = teams_data[side].get("players", {})
            lineup = []
            for idx, pid in enumerate(batting_ids, start=1):
                pkey = f"ID{pid}"
                person = players.get(pkey, {}).get("person", {})
                pos    = players.get(pkey, {}).get("position", {})
                lineup.append({
                    "batting_order": idx,
                    "player_id": pid,
                    "player_name": person.get("fullName", ""),
                    "position": pos.get("abbreviation", ""),
                })
            return lineup
    return []


# ---------------------------------------------------------------------------
# Tier 3: Roster fallback
# ---------------------------------------------------------------------------


def _build_roster_lineup(team_id: int, game_date: date, season: int) -> list[dict]:
    """
    Build a lineup from the active roster sorted by xwOBA (position players only).
    Excludes pitchers. Returns top 9 by season xwOBA.
    """
    roster = fetch_roster(team_id, game_date)
    active = roster.get("active", [])

    # Filter to position players (exclude P, SP, RP)
    batters = [
        p for p in active
        if p.get("position", "") not in ("P", "SP", "RP", "")
        and p.get("player_id") is not None
    ]

    if not batters:
        # Last resort: use all active players
        batters = [p for p in active if p.get("player_id") is not None]

    # Fetch xwOBA for sorting
    batter_xwobas = []
    for p in batters:
        stats = fetch_batter_stats(p["player_id"], season)
        xwoba = stats.get("xwoba", 0.320) if stats else 0.320
        batter_xwobas.append((p, xwoba))

    # Sort by xwOBA descending (best hitters first)
    batter_xwobas.sort(key=lambda x: x[1], reverse=True)

    lineup = []
    for idx, (p, xwoba) in enumerate(batter_xwobas[:9], start=1):
        lineup.append({
            "batting_order": idx,
            "player_id": p["player_id"],
            "player_name": p.get("full_name", f"Player {p['player_id']}"),
            "position": p.get("position", ""),
            "is_confirmed": False,
            "source": "roster",
        })

    return lineup


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_side(confirmed_data: dict, team_id: int, team_abbr: str) -> str:
    """Determine if this team is 'home' or 'away' from confirmed_data."""
    home_tid = confirmed_data.get("home", {}).get("team_id")
    if home_tid == team_id:
        return "home"
    # Also check abbreviation
    home_abbr = confirmed_data.get("home", {}).get("team_abbr", "").upper()
    if home_abbr == team_abbr.upper():
        return "home"
    return "away"


def _persist_lineup(game_pk: int, team_id: int, team_abbr: str,
                    lineup: list[dict], report_type: str) -> None:
    try:
        save_lineups(game_pk, team_id, team_abbr, lineup, report_type)
    except Exception as exc:
        logger.warning("Could not persist lineup to DB: %s", exc)
