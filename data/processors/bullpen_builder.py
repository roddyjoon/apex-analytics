"""
Apex Analytics — Bullpen Profile Builder
Aggregates team bullpen quality from Statcast data.
Handles prior-game fatigue detection (extra innings flag).
"""

import logging
from datetime import date
from typing import Optional

import requests

from config import (
    EXTRA_INNINGS_THRESHOLD,
    BULLPEN_FATIGUE_XFIP_ADJ,
    SEPTEMBER_ROSTER_SIZE,
    SEPTEMBER_BULLPEN_IP_BONUS,
    LEAGUE_AVG_ERA,
)
from data.ingestors.mlb_results import get_prior_night_max_innings
from data.cache.file_cache import get_cache, set_cache
from simulation.profiles import BullpenProfile

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
BULLPEN_CACHE_TTL_HOURS = 6.0


def build_bullpen_profile(
    team_id:    int,
    team_abbr:  str,
    game_date:  Optional[date] = None,
    is_home:    bool = False,
) -> BullpenProfile:
    """
    Build a complete BullpenProfile for a team.

    Steps:
      1. Fetch bullpen xFIP from Baseball Savant pitcher leaderboard
      2. Check prior-game innings for fatigue flag
      3. Compute available innings (season phase adjustment)

    Returns a BullpenProfile with all fields populated.
    """
    if game_date is None:
        game_date = date.today()

    cache_key = f"bullpen_{team_id}_{game_date.isoformat()}"
    cached = get_cache(cache_key)
    # Only use cache hit if it deserialized back as a dict (JSON-safe round-trip)
    if isinstance(cached, dict):
        try:
            return BullpenProfile(
                team_id=cached.get("team_id", team_id),
                team_abbr=cached.get("team_abbr", team_abbr),
                xfip=cached.get("xfip", 4.20),
                high_lev_xfip=cached.get("high_lev_xfip", 4.00),
                era=cached.get("era", 4.20),
                ip_available=cached.get("ip_available", 4.0),
                fatigue_flag=cached.get("fatigue_flag", False),
                prior_game_innings=cached.get("prior_game_innings", 9),
                pct_righties=cached.get("pct_righties", 0.60),
                pct_lefties=cached.get("pct_lefties", 0.40),
            )
        except Exception:
            pass

    # 1. Fetch bullpen aggregate stats
    xfip, high_lev_xfip, era, pct_righties = _fetch_bullpen_stats(team_id, game_date)

    # 2. Check prior-game fatigue
    prior_innings = _get_prior_game_innings(team_id, game_date)
    fatigue_flag  = (prior_innings is not None and prior_innings >= EXTRA_INNINGS_THRESHOLD)

    if fatigue_flag:
        xfip         += BULLPEN_FATIGUE_XFIP_ADJ
        high_lev_xfip += BULLPEN_FATIGUE_XFIP_ADJ
        logger.info(
            "Bullpen fatigue flag: %s played %d innings yesterday. xFIP +%.2f.",
            team_abbr, prior_innings, BULLPEN_FATIGUE_XFIP_ADJ
        )

    # 3. Estimated available innings
    is_september = game_date.month >= 9
    base_ip = 4.0
    if is_september:
        base_ip += SEPTEMBER_BULLPEN_IP_BONUS

    profile = BullpenProfile(
        team_id=team_id,
        team_abbr=team_abbr,
        xfip=round(xfip, 3),
        high_lev_xfip=round(high_lev_xfip, 3),
        era=round(era, 3),
        ip_available=base_ip,
        fatigue_flag=fatigue_flag,
        prior_game_innings=prior_innings or 9,
        pct_righties=pct_righties,
        pct_lefties=round(1.0 - pct_righties, 2),
    )

    # Cache as a plain dict so JSON round-trip works correctly
    set_cache(cache_key, profile.__dict__, ttl_hours=BULLPEN_CACHE_TTL_HOURS)
    return profile


def _apply_fatigue_flag(prior_day_innings: Optional[int]) -> bool:
    """
    Return True if prior-day innings count triggers bullpen fatigue.
    Extracts the inline fatigue check from build_bullpen_profile for testability.
    """
    return prior_day_innings is not None and prior_day_innings >= EXTRA_INNINGS_THRESHOLD


def _compute_ip_available(game_date: date, base_ip: float = 4.0) -> float:
    """
    Return estimated bullpen IP available for a given game date.
    Adds SEPTEMBER_BULLPEN_IP_BONUS during the expanded September roster period.
    """
    if game_date.month >= 9:
        return base_ip + SEPTEMBER_BULLPEN_IP_BONUS
    return base_ip


def _fetch_bullpen_stats(team_id: int, game_date: date) -> tuple[float, float, float, float]:
    """
    Fetch bullpen xFIP, high-leverage xFIP, ERA, and handedness mix from Savant.
    Returns (xfip, high_lev_xfip, era, pct_righties).
    Falls back to league average on failure.
    """
    season = game_date.year

    # Baseball Savant reliever leaderboard (pitcher_type=RP, min IP=10)
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=pitcher&filter=&sort=4&sortDir=desc"
        f"&min=10&selections=xera,xfip,era&chart=false"
        f"&playerType=RP&team={team_id}"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        players = data.get("leaderboard", data.get("data", []))
        if not players:
            return _fallback_bullpen_stats()

        # Aggregate across all team relievers
        xfip_vals  = []
        era_vals   = []
        righty_cnt = 0

        for p in players:
            x = _safe_float(p.get("xfip") or p.get("xFIP"))
            e = _safe_float(p.get("era")  or p.get("ERA"))
            if x is not None and 2.0 <= x <= 7.0:
                xfip_vals.append(x)
            if e is not None and 0.0 <= e <= 10.0:
                era_vals.append(e)
            throws = str(p.get("p_throws", p.get("throws", "R"))).upper()
            if throws == "R":
                righty_cnt += 1

        if not xfip_vals:
            return _fallback_bullpen_stats()

        xfip    = sum(xfip_vals) / len(xfip_vals)
        era     = sum(era_vals) / len(era_vals) if era_vals else xfip
        # High-leverage approximation: top third of relievers by xFIP quality
        sorted_xfip = sorted(xfip_vals)
        top_third   = sorted_xfip[:max(1, len(sorted_xfip)//3)]
        high_lev    = sum(top_third) / len(top_third)
        pct_right   = righty_cnt / max(len(players), 1)

        return xfip, high_lev, era, min(max(pct_right, 0.0), 1.0)

    except Exception as exc:
        logger.warning("Bullpen stats fetch failed for team %d: %s", team_id, exc)
        return _fallback_bullpen_stats()


def _get_prior_game_innings(team_id: int, game_date: date) -> Optional[int]:
    """Check how many innings the team played in their most recent game."""
    try:
        return get_prior_night_max_innings(team_id, game_date)
    except Exception as exc:
        logger.debug("Could not fetch prior innings for team %d: %s", team_id, exc)
        return None


def _fallback_bullpen_stats() -> tuple[float, float, float, float]:
    """League-average fallback: (xFIP, high_lev_xFIP, ERA, pct_righties)."""
    return LEAGUE_AVG_ERA, LEAGUE_AVG_ERA - 0.20, LEAGUE_AVG_ERA, 0.62


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None
