"""
Apex Analytics — Statcast Pitcher Ingestor
Season-to-date aggregates for starting pitchers via pybaseball / Baseball Savant.
Computes xERA, FIP components, SIERA inputs, SwStr%, CSW%, Barrel% allowed,
xBA allowed, home/away ERA splits, days rest, and a per-start game log.
All data is public (no API key). 1-day Statcast delay is expected and handled.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DECAY_WINDOW_DAYS,
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_K_RATE,
    LEAGUE_AVG_BB_RATE,
    LEAGUE_AVG_BARREL_PCT,
    LEAGUE_AVG_SWSTR_PCT,
    LEAGUE_AVG_CSW_PCT,
    LEAGUE_AVG_XWOBA,
    LOW_CONFIDENCE_BF_THRESH,
)
from data.cache.db import save_player_stats, get_player_stats
from data.cache.file_cache import (
    get_cache, set_cache, statcast_cache_key, STATCAST_CACHE_TTL_HOURS,
)

logger = logging.getLogger(__name__)

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


def fetch_pitcher_stats(
    player_id: int,
    season: int,
    team_id: Optional[int] = None,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch season-to-date Statcast aggregates for a pitcher.

    Parameters
    ----------
    player_id : MLB Stats API person ID for the pitcher.
    season    : Season year (e.g. 2025).
    team_id   : Optional team ID used to compute home/away ERA splits.
    force_refresh : Bypass all caches and re-fetch from Statcast.

    Returns
    -------
    dict with keys:
      player_id, season, xera, fip_components, siera_inputs,
      swstr_pct, csw_pct, barrel_pct_allowed, xba_allowed,
      home_era_adj, away_era_adj, bf, throws, confidence, game_log
    """
    cache_key = statcast_cache_key("pitcher", player_id, season)

    if not force_refresh:
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

        db_cached = get_player_stats(player_id, season, "pitcher")
        if db_cached is not None:
            result = db_cached["stats"]
            result["game_log"] = db_cached["game_log"]
            set_cache(cache_key, result, ttl_hours=STATCAST_CACHE_TTL_HOURS)
            return result

    _load_pybaseball()
    import pybaseball

    season_start = f"{season}-03-01"
    season_end = (date.today() - timedelta(days=1)).isoformat()

    try:
        df = pybaseball.statcast_pitcher(season_start, season_end, player_id=player_id)
    except Exception as exc:
        logger.warning(
            "Statcast pitcher fetch failed for player %d (%d): %s",
            player_id, season, exc,
        )
        return _league_average_pitcher(player_id, season)

    if df is None or df.empty:
        logger.debug(
            "No Statcast data for pitcher %d in %d. Using league avg.",
            player_id, season,
        )
        return _league_average_pitcher(player_id, season)

    try:
        stats = _aggregate_pitcher_season(df, player_id, season, team_id)
        game_log = _build_pitcher_game_log(df)
    except Exception as exc:
        logger.error("Pitcher aggregation failed for player %d: %s", player_id, exc)
        return _league_average_pitcher(player_id, season)

    result = {**stats, "game_log": game_log}

    save_player_stats(player_id, season, "pitcher", stats, game_log)
    set_cache(cache_key, result, ttl_hours=STATCAST_CACHE_TTL_HOURS)

    logger.debug(
        "Fetched pitcher stats for player %d: BF=%d, xERA=%.2f",
        player_id, stats.get("bf", 0), stats.get("xera", LEAGUE_AVG_ERA),
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _aggregate_pitcher_season(
    df: pd.DataFrame,
    player_id: int,
    season: int,
    team_id: Optional[int] = None,
) -> dict:
    """Compute season aggregates from raw Statcast pitch-level DataFrame."""
    total_pitches = len(df)
    if total_pitches == 0:
        return _league_average_pitcher(player_id, season)

    # Unique PA proxy (batters faced): group by game_date + batter
    if "batter" in df.columns and "game_date" in df.columns:
        bf = int(df.groupby(["game_date", "batter"]).ngroups)
    else:
        bf = total_pitches // 4  # rough fallback

    # Throws hand
    throws = "R"
    if "pitcher_1_throws" in df.columns:
        mode_val = df["pitcher_1_throws"].mode()
        throws = str(mode_val.iloc[0]) if not mode_val.empty else "R"
    elif "p_throws" in df.columns:
        mode_val = df["p_throws"].mode()
        throws = str(mode_val.iloc[0]) if not mode_val.empty else "R"

    # --- xERA (from estimated_woba_using_speedangle as proxy, or p_era column) ---
    xera = LEAGUE_AVG_ERA
    if "estimated_woba_using_speedangle" in df.columns:
        xwoba_vals = df["estimated_woba_using_speedangle"].dropna()
        if len(xwoba_vals) > 0:
            # Convert xwOBA to ERA proxy using linear scaling (league avg xwOBA ~ 0.320 = 4.20 ERA)
            xwoba_mean = float(xwoba_vals.mean())
            xera = round(LEAGUE_AVG_ERA * (xwoba_mean / max(LEAGUE_AVG_XWOBA, 0.001)), 2)
            xera = max(1.00, min(xera, 12.00))

    # --- FIP components ---
    events_col = "events"
    k_count = 0
    bb_count = 0
    hr_count = 0
    if events_col in df.columns:
        k_count = int((df[events_col] == "strikeout").sum())
        bb_count = int(df[events_col].isin(["walk", "intent_walk"]).sum())
        hr_count = int((df[events_col] == "home_run").sum())

    k_pct = k_count / bf if bf > 0 else LEAGUE_AVG_K_RATE
    bb_pct = bb_count / bf if bf > 0 else LEAGUE_AVG_BB_RATE
    hr_pct_per_9 = (hr_count * 9.0 / (bf / 3.0)) if bf > 0 else (LEAGUE_AVG_ERA * 0.03)

    fip_components = {
        "k_pct": round(k_pct, 4),
        "bb_pct": round(bb_pct, 4),
        "hr_pct_per_9": round(hr_pct_per_9, 4),
    }

    # --- SIERA inputs (batted ball rates) ---
    gb_pct = fb_pct = ld_pct = 0.0
    if "bb_type" in df.columns:
        batted = df["bb_type"].dropna()
        total_bb = len(batted)
        if total_bb > 0:
            gb_pct = float((batted == "ground_ball").sum() / total_bb)
            fb_pct = float((batted == "fly_ball").sum() / total_bb)
            ld_pct = float((batted == "line_drive").sum() / total_bb)
    else:
        # Default league-average batted ball rates
        gb_pct, fb_pct, ld_pct = 0.44, 0.35, 0.21

    siera_inputs = {
        "k_pct": round(k_pct, 4),
        "bb_pct": round(bb_pct, 4),
        "gb_pct": round(gb_pct, 4),
        "fb_pct": round(fb_pct, 4),
        "ld_pct": round(ld_pct, 4),
    }

    # --- SwStr% and CSW% ---
    swstr_pct = LEAGUE_AVG_SWSTR_PCT
    csw_pct = LEAGUE_AVG_CSW_PCT
    if "description" in df.columns:
        swinging_strikes = int(
            df["description"].str.contains("swinging_strike", na=False).sum()
        )
        called_strikes = int(
            df["description"].str.contains("called_strike", na=False).sum()
        )
        swstr_pct = swinging_strikes / total_pitches if total_pitches > 0 else LEAGUE_AVG_SWSTR_PCT
        csw_pct = (called_strikes + swinging_strikes) / total_pitches if total_pitches > 0 else LEAGUE_AVG_CSW_PCT
        swstr_pct = round(float(swstr_pct), 4)
        csw_pct = round(float(csw_pct), 4)

    # --- Barrel% allowed ---
    barrel_pct_allowed = LEAGUE_AVG_BARREL_PCT
    if "barrel" in df.columns:
        barrels = df["barrel"].fillna(0).sum()
        batted_balls = df["barrel"].notna().sum()
        if batted_balls > 0:
            barrel_pct_allowed = round(float(barrels / batted_balls), 4)

    # --- xBA allowed ---
    xba_allowed = 0.250
    if "estimated_ba_using_speedangle" in df.columns:
        xba_vals = df["estimated_ba_using_speedangle"].dropna()
        if len(xba_vals) > 0:
            xba_allowed = round(float(xba_vals.mean()), 3)

    # --- Home/Away ERA split multipliers ---
    home_era_adj, away_era_adj = _compute_home_away_era_adj(df, team_id)

    # --- Days rest (from last game date) ---
    days_rest = _compute_days_rest(df)

    # --- Confidence ---
    if bf >= 300:
        confidence = "HIGH"
    elif bf >= LOW_CONFIDENCE_BF_THRESH:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "player_id": player_id,
        "season": season,
        "xera": xera,
        "fip_components": fip_components,
        "siera_inputs": siera_inputs,
        "swstr_pct": swstr_pct,
        "csw_pct": csw_pct,
        "barrel_pct_allowed": barrel_pct_allowed,
        "xba_allowed": xba_allowed,
        "home_era_adj": home_era_adj,
        "away_era_adj": away_era_adj,
        "days_rest": days_rest,
        "bf": bf,
        "throws": throws,
        "confidence": confidence,
    }


def _compute_home_away_era_adj(
    df: pd.DataFrame, team_id: Optional[int]
) -> tuple[float, float]:
    """
    Compute multiplicative ERA adjustment factors for home vs. away contexts.
    Returns (home_adj, away_adj) where 1.0 = league-neutral performance.
    If team_id is None or splits cannot be computed, returns (1.0, 1.0).
    """
    if team_id is None or "home_team" not in df.columns:
        return 1.0, 1.0

    if "estimated_woba_using_speedangle" not in df.columns:
        return 1.0, 1.0

    try:
        home_df = df[df["home_team"] == team_id]
        away_df = df[df["home_team"] != team_id]

        overall_xwoba = df["estimated_woba_using_speedangle"].dropna().mean()
        if pd.isna(overall_xwoba) or overall_xwoba == 0:
            return 1.0, 1.0

        home_xwoba = home_df["estimated_woba_using_speedangle"].dropna().mean()
        away_xwoba = away_df["estimated_woba_using_speedangle"].dropna().mean()

        home_adj = float(home_xwoba / overall_xwoba) if not pd.isna(home_xwoba) and overall_xwoba > 0 else 1.0
        away_adj = float(away_xwoba / overall_xwoba) if not pd.isna(away_xwoba) and overall_xwoba > 0 else 1.0

        # Clamp to sensible range
        home_adj = max(0.70, min(home_adj, 1.40))
        away_adj = max(0.70, min(away_adj, 1.40))

        return round(home_adj, 3), round(away_adj, 3)
    except Exception as exc:
        logger.debug("Home/away ERA split computation failed: %s", exc)
        return 1.0, 1.0


def _compute_days_rest(df: pd.DataFrame) -> Optional[int]:
    """Return days since the pitcher's last outing, or None if unknown."""
    if "game_date" not in df.columns:
        return None
    try:
        df_copy = df.copy()
        df_copy["game_date"] = pd.to_datetime(df_copy["game_date"])
        game_dates = df_copy["game_date"].dropna().sort_values()
        if len(game_dates) == 0:
            return None
        last_game = game_dates.iloc[-1].date()
        return (date.today() - last_game).days
    except Exception:
        return None


def _build_pitcher_game_log(df: pd.DataFrame) -> list[dict]:
    """
    Aggregate pitch-level Statcast data into per-start entries.
    Returns list of {date, xera, k_pct, bb_pct, ip, bf} sorted newest-first.
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
        total_pitches = len(group)

        # BF proxy
        if "batter" in group.columns:
            bf = int(group["batter"].nunique())
        else:
            bf = max(1, total_pitches // 4)

        # IP proxy (3 outs per inning, ~4 pitches per PA)
        # Use estimated outs: K + ball-in-play (events that end PA)
        ip = 0.0
        if "events" in group.columns:
            out_events = {"strikeout", "field_out", "grounded_into_double_play",
                          "force_out", "fielders_choice_out", "double_play",
                          "sac_fly", "sac_bunt"}
            outs = int(group["events"].isin(out_events).sum())
            ip = round(outs / 3.0, 1)

        # xERA for this start
        xwoba_vals = group.get("estimated_woba_using_speedangle", pd.Series(dtype=float)).dropna()
        xwoba_mean = float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else LEAGUE_AVG_XWOBA
        start_xera = round(LEAGUE_AVG_ERA * (xwoba_mean / max(LEAGUE_AVG_XWOBA, 0.001)), 2)
        start_xera = max(0.0, min(start_xera, 12.0))

        # K% and BB%
        if "events" in group.columns:
            k_count = int((group["events"] == "strikeout").sum())
            bb_count = int(group["events"].isin(["walk", "intent_walk"]).sum())
        else:
            k_count = bb_count = 0
        k_pct = round(k_count / bf, 4) if bf > 0 else 0.0
        bb_pct = round(bb_count / bf, 4) if bf > 0 else 0.0

        game_log.append({
            "date": game_date.date().isoformat(),
            "xera": start_xera,
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "ip": ip,
            "bf": bf,
        })

    game_log.sort(key=lambda x: x["date"], reverse=True)
    return game_log


def _league_average_pitcher(
    player_id: Optional[int] = None,
    season: Optional[int] = None,
) -> dict:
    """Return league-average pitcher profile for fallback."""
    return {
        "player_id": player_id,
        "season": season,
        "xera": LEAGUE_AVG_ERA,
        "fip_components": {
            "k_pct": LEAGUE_AVG_K_RATE,
            "bb_pct": LEAGUE_AVG_BB_RATE,
            "hr_pct_per_9": 1.25,
        },
        "siera_inputs": {
            "k_pct": LEAGUE_AVG_K_RATE,
            "bb_pct": LEAGUE_AVG_BB_RATE,
            "gb_pct": 0.44,
            "fb_pct": 0.35,
            "ld_pct": 0.21,
        },
        "swstr_pct": LEAGUE_AVG_SWSTR_PCT,
        "csw_pct": LEAGUE_AVG_CSW_PCT,
        "barrel_pct_allowed": LEAGUE_AVG_BARREL_PCT,
        "xba_allowed": 0.250,
        "home_era_adj": 1.0,
        "away_era_adj": 1.0,
        "days_rest": None,
        "bf": 0,
        "throws": "R",
        "confidence": "LOW",
        "game_log": [],
    }
