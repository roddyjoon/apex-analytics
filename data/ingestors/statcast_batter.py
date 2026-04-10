"""
Apex Analytics — Statcast Batter Ingestor
Season-to-date aggregates + game-by-game log via pybaseball / Baseball Savant.
All data is public (no API key). 1-day delay on Statcast data is expected and handled.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DECAY_WINDOW_DAYS,
    LEAGUE_AVG_XWOBA,
    LEAGUE_AVG_BARREL_PCT,
    LEAGUE_AVG_SWSTR_PCT,
    LEAGUE_AVG_K_RATE,
    LEAGUE_AVG_BB_RATE,
    LEAGUE_AVG_SPRINT_SPEED,
)
from data.cache.db import save_player_stats, get_player_stats
from data.cache.file_cache import (
    get_cache, set_cache, statcast_cache_key, STATCAST_CACHE_TTL_HOURS
)

logger = logging.getLogger(__name__)

# Lazy import pybaseball to avoid slow startup when not needed
_pybaseball_loaded = False


def _load_pybaseball():
    global _pybaseball_loaded
    if not _pybaseball_loaded:
        try:
            import pybaseball
            pybaseball.cache.enable()
            _pybaseball_loaded = True
        except ImportError:
            raise ImportError("pybaseball is required. Run: pip install pybaseball")


def fetch_sprint_speed(player_id: int, season: int) -> float:
    """
    Fetch sprint speed for a batter from the Baseball Savant sprint speed leaderboard.

    Uses pybaseball.statcast_sprint_speed(season) which returns a leaderboard
    DataFrame with columns including player_id and sprint_speed.

    Falls back to LEAGUE_AVG_SPRINT_SPEED (27.0) if player not found or on error.
    Result is cached for 24 hours.

    Parameters
    ----------
    player_id : MLB player ID (e.g. 592450 for Mike Trout)
    season    : Year (e.g. 2025)

    Returns
    -------
    float — sprint speed in ft/s, or 27.0 (league avg) on failure.
    """
    cache_key = f"sprint_speed_{player_id}_{season}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    _load_pybaseball()
    import pybaseball

    try:
        # statcast_sprint_speed takes year as string, optional min_opp/position/team
        df = pybaseball.statcast_sprint_speed(str(season))
    except Exception as exc:
        logger.debug(
            "Sprint speed leaderboard fetch failed for season %d: %s — using league avg",
            season, exc,
        )
        return LEAGUE_AVG_SPRINT_SPEED

    if df is None or df.empty:
        logger.debug("Empty sprint speed leaderboard for %d — using league avg.", season)
        return LEAGUE_AVG_SPRINT_SPEED

    # Normalise column names: pybaseball may use 'player_id' or 'IDfg' etc.
    # The sprint speed leaderboard typically has 'player_id' and 'sprint_speed'
    player_id_col = None
    for col in ("player_id", "IDfg", "mlbamid", "bam_id"):
        if col in df.columns:
            player_id_col = col
            break

    if player_id_col is None:
        logger.debug(
            "Sprint speed leaderboard has no recognised player_id column "
            "(columns: %s) — using league avg.", list(df.columns)
        )
        return LEAGUE_AVG_SPRINT_SPEED

    speed_col = None
    for col in ("sprint_speed", "r_sprint_speed_top50p"):
        if col in df.columns:
            speed_col = col
            break

    if speed_col is None:
        logger.debug(
            "Sprint speed leaderboard has no recognised speed column "
            "(columns: %s) — using league avg.", list(df.columns)
        )
        return LEAGUE_AVG_SPRINT_SPEED

    try:
        # player_id may be stored as int or float; cast both sides for safety
        row = df[df[player_id_col].astype(int) == int(player_id)]
        if row.empty:
            logger.debug(
                "Player %d not found in sprint speed leaderboard for %d — using league avg.",
                player_id, season,
            )
            result = LEAGUE_AVG_SPRINT_SPEED
        else:
            speed_val = row.iloc[0][speed_col]
            result = float(speed_val) if pd.notna(speed_val) else LEAGUE_AVG_SPRINT_SPEED
            logger.debug(
                "Sprint speed for player %d (%d): %.1f ft/s", player_id, season, result
            )
    except Exception as exc:
        logger.debug("Sprint speed lookup failed for player %d: %s — using league avg",
                     player_id, exc)
        result = LEAGUE_AVG_SPRINT_SPEED

    # Cache for 24 hours (leaderboard is updated daily)
    set_cache(cache_key, result, ttl_hours=24)
    return result


def fetch_batter_stats(player_id: int, season: int,
                       force_refresh: bool = False) -> Optional[dict]:
    """
    Fetch season-to-date Statcast aggregates for a batter.

    Returns a dict with keys:
      xwoba, xba, barrel_pct, hard_hit_pct, swstr_pct, k_pct, bb_pct,
      hr_rate, sprint_speed, obp, slg, pa, bats (L/R/S),
      xwoba_vs_lhp, xwoba_vs_rhp,
      game_log: list of {date, xwoba, pa, hr, k, bb} per game
    Returns league averages if player data not found.
    """
    cache_key = statcast_cache_key("batter", player_id, season)

    if not force_refresh:
        # Check in-memory file cache first
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

        # Check DB cache
        db_cached = get_player_stats(player_id, season, "batter")
        if db_cached is not None:
            # Re-wrap for consistent return format
            result = db_cached["stats"]
            result["game_log"] = db_cached["game_log"]
            set_cache(cache_key, result, ttl_hours=STATCAST_CACHE_TTL_HOURS)
            return result

    _load_pybaseball()
    import pybaseball

    season_start = f"{season}-03-01"
    # Statcast has 1-day delay; use yesterday as end date
    season_end = (date.today() - timedelta(days=1)).isoformat()

    try:
        df = pybaseball.statcast_batter(season_start, season_end, player_id=player_id)
    except Exception as exc:
        logger.warning("Statcast batter fetch failed for player %d (%d): %s",
                       player_id, season, exc)
        return _league_average_batter()

    if df is None or df.empty:
        logger.debug("No Statcast data for batter %d in %d. Using league avg.", player_id, season)
        return _league_average_batter()

    try:
        stats = _aggregate_batter_season(df, player_id, season)
        game_log = _build_batter_game_log(df)
    except Exception as exc:
        logger.error("Batter aggregation failed for player %d: %s", player_id, exc)
        return _league_average_batter()

    # Fetch real sprint speed from the leaderboard (separate pull, cached 24h)
    stats["sprint_speed"] = fetch_sprint_speed(player_id, season)

    result = {**stats, "game_log": game_log}

    # Persist to DB and file cache
    save_player_stats(player_id, season, "batter", stats, game_log)
    set_cache(cache_key, result, ttl_hours=STATCAST_CACHE_TTL_HOURS)

    logger.debug("Fetched batter stats for player %d: PA=%d, xwOBA=%.3f",
                 player_id, stats.get("pa", 0), stats.get("xwoba", 0))
    return result


def fetch_batter_platoon_splits(player_id: int, season: int) -> dict:
    """
    Return xwOBA splits vs. LHP and vs. RHP from Statcast game data.
    Falls back to overall xwOBA if insufficient sample.
    """
    cache_key = f"batter_platoon_{player_id}_{season}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    _load_pybaseball()
    import pybaseball

    season_start = f"{season}-03-01"
    season_end = (date.today() - timedelta(days=1)).isoformat()

    try:
        df = pybaseball.statcast_batter(season_start, season_end, player_id=player_id)
    except Exception:
        return {"xwoba_vs_lhp": LEAGUE_AVG_XWOBA, "xwoba_vs_rhp": LEAGUE_AVG_XWOBA,
                "pa_vs_lhp": 0, "pa_vs_rhp": 0}

    if df is None or df.empty:
        return {"xwoba_vs_lhp": LEAGUE_AVG_XWOBA, "xwoba_vs_rhp": LEAGUE_AVG_XWOBA,
                "pa_vs_lhp": 0, "pa_vs_rhp": 0}

    # Statcast column: p_throws (L or R)
    splits = {}
    for hand, label in [("L", "lhp"), ("R", "rhp")]:
        subset = df[df["p_throws"] == hand] if "p_throws" in df.columns else pd.DataFrame()
        if subset.empty or "estimated_woba_using_speedangle" not in subset.columns:
            splits[f"xwoba_vs_{label}"] = LEAGUE_AVG_XWOBA
            splits[f"pa_vs_{label}"] = 0
        else:
            xwoba_vals = subset["estimated_woba_using_speedangle"].dropna()
            splits[f"xwoba_vs_{label}"] = float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else LEAGUE_AVG_XWOBA
            splits[f"pa_vs_{label}"] = len(subset)

    set_cache(cache_key, splits, ttl_hours=STATCAST_CACHE_TTL_HOURS)
    return splits


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _aggregate_batter_season(df: "pd.DataFrame", player_id: int, season: int) -> dict:
    """Compute season aggregates from raw Statcast PA-level DataFrame."""
    total_pa = len(df)
    if total_pa == 0:
        return _league_average_batter()

    # xwOBA — estimated_woba_using_speedangle
    xwoba_col = "estimated_woba_using_speedangle"
    xwoba_vals = df[xwoba_col].dropna() if xwoba_col in df.columns else pd.Series(dtype=float)
    xwoba = float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else LEAGUE_AVG_XWOBA

    # xBA — estimated_ba_using_speedangle
    xba_col = "estimated_ba_using_speedangle"
    xba_vals = df[xba_col].dropna() if xba_col in df.columns else pd.Series(dtype=float)
    xba = float(xba_vals.mean()) if len(xba_vals) > 0 else 0.250

    # Barrel% — is_barrel flag
    barrel_col = "barrel"
    if barrel_col in df.columns:
        barrels = df[barrel_col].fillna(0).sum()
        batted_balls = df[barrel_col].notna().sum()
        barrel_pct = float(barrels / batted_balls) if batted_balls > 0 else LEAGUE_AVG_BARREL_PCT
    else:
        barrel_pct = LEAGUE_AVG_BARREL_PCT

    # Hard-Hit% — exit_velocity >= 95
    ev_col = "launch_speed"
    if ev_col in df.columns:
        ev_vals = df[ev_col].dropna()
        hard_hit_pct = float((ev_vals >= 95).sum() / len(ev_vals)) if len(ev_vals) > 0 else 0.38
    else:
        hard_hit_pct = 0.38

    # SwStr% — description contains "swinging_strike"
    desc_col = "description"
    if desc_col in df.columns:
        swinging_strikes = df[desc_col].str.contains("swinging_strike", na=False).sum()
        swstr_pct = float(swinging_strikes / total_pa) if total_pa > 0 else LEAGUE_AVG_SWSTR_PCT
    else:
        swstr_pct = LEAGUE_AVG_SWSTR_PCT

    # K% — events == "strikeout"
    events_col = "events"
    if events_col in df.columns:
        k_count = (df[events_col] == "strikeout").sum()
        k_pct = float(k_count / total_pa)
        bb_count = df[events_col].isin(["walk", "intent_walk"]).sum()
        bb_pct = float(bb_count / total_pa)
        hr_count = (df[events_col] == "home_run").sum()
        hr_rate = float(hr_count / total_pa)
    else:
        k_pct = LEAGUE_AVG_K_RATE
        bb_pct = LEAGUE_AVG_BB_RATE
        hr_rate = 0.037
        hr_count = 0

    # Sprint speed — fetched separately via fetch_sprint_speed() and injected
    # by fetch_batter_stats() after this aggregation step; default here is
    # overridden immediately, so this value is never used in practice.
    sprint_speed = LEAGUE_AVG_SPRINT_SPEED

    # OBP / SLG proxy from xwOBA (not exact but ballpark)
    obp = min(xwoba + 0.030, 0.999)
    slg = max(xwoba * 1.3, 0.200)

    return {
        "player_id": player_id,
        "season": season,
        "pa": total_pa,
        "xwoba": xwoba,
        "xba": xba,
        "barrel_pct": barrel_pct,
        "hard_hit_pct": hard_hit_pct,
        "swstr_pct": swstr_pct,
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "hr_rate": hr_rate,
        "sprint_speed": sprint_speed,
        "obp": obp,
        "slg": slg,
        "confidence": "HIGH" if total_pa >= 100 else ("MEDIUM" if total_pa >= 50 else "LOW"),
    }


def _build_batter_game_log(df: "pd.DataFrame") -> list[dict]:
    """
    Aggregate Statcast PA-level data into per-game entries for recency weighting.
    Returns list of {date, xwoba, pa, hr, k, bb} sorted newest-first.
    """
    if "game_date" not in df.columns:
        return []

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    cutoff = pd.Timestamp(date.today() - timedelta(days=DECAY_WINDOW_DAYS + 5))
    df_recent = df[df["game_date"] >= cutoff].copy()

    if df_recent.empty:
        return []

    game_log = []
    for game_date, group in df_recent.groupby("game_date"):
        pa = len(group)
        xwoba_vals = group.get("estimated_woba_using_speedangle", pd.Series(dtype=float)).dropna()
        xwoba = float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else LEAGUE_AVG_XWOBA

        events = group.get("events", pd.Series(dtype=str))
        hr = int((events == "home_run").sum())
        k  = int((events == "strikeout").sum())
        bb = int(events.isin(["walk", "intent_walk"]).sum())

        game_log.append({
            "date": game_date.date().isoformat(),
            "xwoba": xwoba,
            "pa": pa,
            "hr": hr,
            "k": k,
            "bb": bb,
        })

    # Sort newest first
    game_log.sort(key=lambda x: x["date"], reverse=True)
    return game_log


def _league_average_batter() -> dict:
    """Return league-average batter profile for fallback."""
    return {
        "player_id": None,
        "season": None,
        "pa": 0,
        "xwoba": LEAGUE_AVG_XWOBA,
        "xba": 0.250,
        "barrel_pct": LEAGUE_AVG_BARREL_PCT,
        "hard_hit_pct": 0.380,
        "swstr_pct": LEAGUE_AVG_SWSTR_PCT,
        "k_pct": LEAGUE_AVG_K_RATE,
        "bb_pct": LEAGUE_AVG_BB_RATE,
        "hr_rate": 0.037,
        "sprint_speed": LEAGUE_AVG_SPRINT_SPEED,
        "obp": 0.315,
        "slg": 0.413,
        "confidence": "LOW",
        "game_log": [],
    }
