"""
Apex Analytics — Pitcher Pitch Arsenal Ingestor
Fetches per-pitch-type breakdown for a pitcher from Baseball Savant CSV export.
Computes usage%, xwOBA conceded, SwStr%, and average velocity per pitch type.
Falls back gracefully to an empty list if data is unavailable.
"""

import io
import logging
from datetime import date
from typing import Optional

import pandas as pd
import requests

from data.cache.db import save_pitcher_arsenal, get_pitcher_arsenal
from data.cache.file_cache import (
    get_cache, set_cache, arsenal_cache_key, STATCAST_CACHE_TTL_HOURS,
)

logger = logging.getLogger(__name__)

SAVANT_CSV_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?hfSea={season}|&player_type=pitcher"
    "&pitchers_lookup[]={player_id}"
    "&type=details"
    "&group_by=name-pitch-type"
)

# Minimum pitches thrown of a given type to include in output
MIN_PITCHES = 50


def fetch_pitch_arsenal(
    player_id: int,
    season: int,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch pitch arsenal breakdown for a pitcher from Baseball Savant.

    Returns a list of dicts, one per pitch type:
      [{pitch_type, usage_pct, xwoba_conceded, swstr_pct, avg_velocity}, ...]

    Returns an empty list if data is unavailable or the request fails.
    """
    cache_key = arsenal_cache_key(player_id, season)

    if not force_refresh:
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

        db_cached = get_pitcher_arsenal(player_id, season)
        if db_cached:
            set_cache(cache_key, db_cached, ttl_hours=STATCAST_CACHE_TTL_HOURS)
            return db_cached

    url = SAVANT_CSV_URL.format(season=season, player_id=player_id)

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        raw_text = resp.text
    except requests.RequestException as exc:
        logger.warning(
            "Pitch arsenal CSV fetch failed for player %d (%d): %s",
            player_id, season, exc,
        )
        return []

    if not raw_text or raw_text.strip() == "" or "pitch_type" not in raw_text:
        logger.debug(
            "No pitch arsenal data returned for player %d (%d).", player_id, season
        )
        return []

    try:
        df = pd.read_csv(io.StringIO(raw_text), low_memory=False)
    except Exception as exc:
        logger.warning(
            "Failed to parse pitch arsenal CSV for player %d: %s", player_id, exc
        )
        return []

    if df.empty or "pitch_type" not in df.columns:
        return []

    arsenal = _aggregate_arsenal(df)

    if arsenal:
        save_pitcher_arsenal(player_id, season, arsenal)
        set_cache(cache_key, arsenal, ttl_hours=STATCAST_CACHE_TTL_HOURS)
        logger.debug(
            "Fetched pitch arsenal for player %d: %d pitch types.",
            player_id, len(arsenal),
        )

    return arsenal


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _aggregate_arsenal(df: pd.DataFrame) -> list[dict]:
    """
    Group raw pitch-level DataFrame by pitch_type and compute per-type metrics.
    Filters to pitch types with >= MIN_PITCHES thrown.
    """
    df = df.copy()

    # Normalise pitch_type: drop nulls and empty strings
    if "pitch_type" not in df.columns:
        return []
    df = df[df["pitch_type"].notna() & (df["pitch_type"].astype(str).str.strip() != "")]

    total_pitches = len(df)
    if total_pitches == 0:
        return []

    arsenal = []
    for pitch_type, group in df.groupby("pitch_type"):
        pitch_count = len(group)
        if pitch_count < MIN_PITCHES:
            continue

        # Usage %
        usage_pct = round(pitch_count / total_pitches, 4)

        # xwOBA conceded
        xwoba_col = "estimated_woba_using_speedangle"
        if xwoba_col in group.columns:
            xwoba_vals = group[xwoba_col].dropna()
            xwoba_conceded = round(float(xwoba_vals.mean()), 4) if len(xwoba_vals) > 0 else None
        else:
            xwoba_conceded = None

        # SwStr%
        swstr_pct = None
        if "description" in group.columns:
            swinging = group["description"].str.contains("swinging_strike", na=False).sum()
            swstr_pct = round(float(swinging / pitch_count), 4)

        # Average velocity
        avg_velocity = None
        if "release_speed" in group.columns:
            vel_vals = group["release_speed"].dropna()
            avg_velocity = round(float(vel_vals.mean()), 1) if len(vel_vals) > 0 else None

        arsenal.append({
            "pitch_type": str(pitch_type),
            "usage_pct": usage_pct,
            "xwoba_conceded": xwoba_conceded,
            "swstr_pct": swstr_pct,
            "avg_velocity": avg_velocity,
        })

    # Sort by usage descending
    arsenal.sort(key=lambda x: x["usage_pct"], reverse=True)
    return arsenal
