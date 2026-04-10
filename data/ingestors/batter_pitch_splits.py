"""
Apex Analytics — Batter Pitch-Type Splits Ingestor
Fetches batter xwOBA splits by pitch type from Baseball Savant CSV export.
Used to model pitch-type vulnerability when matched against a pitcher's arsenal.
"""

import io
import logging
from typing import Optional

import pandas as pd
import requests

from data.cache.db import save_pitch_type_splits, get_pitch_type_splits
from data.cache.file_cache import (
    get_cache, set_cache, pitch_splits_cache_key, STATCAST_CACHE_TTL_HOURS,
)

logger = logging.getLogger(__name__)

SAVANT_CSV_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?hfSea={season}|&player_type=batter"
    "&batters_lookup[]={player_id}"
    "&type=details"
    "&group_by=name-pitch-type"
)

# Pitch types we care about for matchup modelling
TRACKED_PITCH_TYPES = {"FF", "SL", "CH", "CU", "SI", "KC", "FC"}

# Minimum PA vs. a given pitch type to include the split
MIN_PA = 30


def fetch_batter_pitch_splits(
    player_id: int,
    season: int,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch batter xwOBA splits by pitch type from Baseball Savant.

    Returns a list of dicts, one per pitch type:
      [{pitch_type, xwoba, ba, k_pct, pa}, ...]

    Only pitch types in TRACKED_PITCH_TYPES with >= MIN_PA plate appearances
    are included. Returns an empty list on error or insufficient data.
    """
    cache_key = pitch_splits_cache_key(player_id, season)

    if not force_refresh:
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

        db_cached = get_pitch_type_splits(player_id, season)
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
            "Batter pitch splits CSV fetch failed for player %d (%d): %s",
            player_id, season, exc,
        )
        return []

    if not raw_text or raw_text.strip() == "" or "pitch_type" not in raw_text:
        logger.debug(
            "No pitch split data returned for batter %d (%d).", player_id, season
        )
        return []

    try:
        df = pd.read_csv(io.StringIO(raw_text), low_memory=False)
    except Exception as exc:
        logger.warning(
            "Failed to parse batter pitch splits CSV for player %d: %s",
            player_id, exc,
        )
        return []

    if df.empty or "pitch_type" not in df.columns:
        return []

    splits = _aggregate_splits(df)

    if splits:
        save_pitch_type_splits(player_id, season, splits)
        set_cache(cache_key, splits, ttl_hours=STATCAST_CACHE_TTL_HOURS)
        logger.debug(
            "Fetched pitch splits for batter %d: %d pitch types.", player_id, len(splits)
        )

    return splits


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _aggregate_splits(df: pd.DataFrame) -> list[dict]:
    """
    Group raw pitch-level DataFrame by pitch_type and compute per-type metrics.
    Filters to TRACKED_PITCH_TYPES with >= MIN_PA plate appearances.
    """
    df = df.copy()

    if "pitch_type" not in df.columns:
        return []

    df = df[
        df["pitch_type"].notna()
        & (df["pitch_type"].astype(str).str.strip() != "")
    ]

    if df.empty:
        return []

    splits = []
    for pitch_type, group in df.groupby("pitch_type"):
        pt_str = str(pitch_type).strip()
        if pt_str not in TRACKED_PITCH_TYPES:
            continue

        pa = len(group)
        if pa < MIN_PA:
            continue

        # xwOBA
        xwoba_col = "estimated_woba_using_speedangle"
        if xwoba_col in group.columns:
            xwoba_vals = group[xwoba_col].dropna()
            xwoba = round(float(xwoba_vals.mean()), 4) if len(xwoba_vals) > 0 else None
        else:
            xwoba = None

        # BA approximation — mean of 'hit' column (1 = hit, 0 = out)
        ba = None
        if "hit" in group.columns:
            hit_vals = group["hit"].dropna()
            ba = round(float(hit_vals.mean()), 3) if len(hit_vals) > 0 else None
        elif "events" in group.columns:
            hit_events = {"single", "double", "triple", "home_run"}
            hit_count = group["events"].isin(hit_events).sum()
            ab_count = group["events"].notna().sum()
            ba = round(float(hit_count / ab_count), 3) if ab_count > 0 else None

        # K%
        k_pct = None
        if "events" in group.columns:
            k_count = (group["events"] == "strikeout").sum()
            k_pct = round(float(k_count / pa), 4)

        splits.append({
            "pitch_type": pt_str,
            "xwoba": xwoba,
            "ba": ba,
            "k_pct": k_pct,
            "pa": pa,
        })

    # Sort by PA descending for readability
    splits.sort(key=lambda x: x["pa"], reverse=True)
    return splits
