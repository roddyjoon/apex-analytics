"""
Apex Analytics — Batter vs. Pitcher Matchup History Ingestor
Fetches career batter-vs-pitcher plate appearance history from Baseball Savant.
Checks DB first, then Savant. Handles rate-limiting with one retry.
Matchups with fewer than 5 PA are discarded (too small a sample to be useful).
"""

import io
import logging
import time
from typing import Optional

import pandas as pd
import requests

from config import MATCHUP_CACHE_REFRESH_DAYS
from data.cache.db import save_matchup_history, get_matchup_history
from data.cache.file_cache import get_cache, set_cache, matchup_cache_key

logger = logging.getLogger(__name__)

SAVANT_CSV_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?hfSea=&player_type=batter"
    "&batters_lookup[]={batter_id}"
    "&pitchers_lookup[]={pitcher_id}"
    "&type=details"
)

# Minimum PA to consider a matchup worth storing / returning
MIN_PA = 5

# TTL for file cache (mirrors config MATCHUP_CACHE_REFRESH_DAYS)
MATCHUP_CACHE_TTL_HOURS = MATCHUP_CACHE_REFRESH_DAYS * 24


def fetch_matchup_history(
    batter_id: int,
    pitcher_id: int,
    force_refresh: bool = False,
) -> Optional[dict]:
    """
    Fetch career batter-vs-pitcher plate appearance history.

    Lookup order:
      1. File cache (fresh within MATCHUP_CACHE_REFRESH_DAYS)
      2. DB cache
      3. Baseball Savant CSV (with one retry on HTTP 429)

    Returns a dict with keys:
      {pa, ab, hits, hr, bb, k, xwoba}
    Returns None if PA < 5 or data unavailable.
    """
    cache_key = matchup_cache_key(batter_id, pitcher_id)

    if not force_refresh:
        # Check fast file cache
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

        # Check DB
        db_cached = get_matchup_history(batter_id, pitcher_id)
        if db_cached is not None:
            result = {
                "pa": db_cached.get("pa", 0),
                "ab": db_cached.get("ab", 0),
                "hits": db_cached.get("hits", 0),
                "hr": db_cached.get("hr", 0),
                "bb": db_cached.get("bb", 0),
                "k": db_cached.get("k", 0),
                "xwoba": db_cached.get("xwoba"),
            }
            set_cache(cache_key, result, ttl_hours=MATCHUP_CACHE_TTL_HOURS)
            return result

    # Fetch from Baseball Savant
    raw_text = _fetch_savant_csv(batter_id, pitcher_id)
    if raw_text is None:
        return None

    if not raw_text or raw_text.strip() == "" or "batter" not in raw_text:
        logger.debug(
            "No matchup history returned for batter %d vs. pitcher %d.",
            batter_id, pitcher_id,
        )
        return None

    try:
        df = pd.read_csv(io.StringIO(raw_text), low_memory=False)
    except Exception as exc:
        logger.warning(
            "Failed to parse matchup CSV for batter %d vs. pitcher %d: %s",
            batter_id, pitcher_id, exc,
        )
        return None

    if df.empty:
        return None

    result = _aggregate_matchup(df)
    if result is None:
        return None

    # Persist
    save_matchup_history(batter_id, pitcher_id, result)
    set_cache(cache_key, result, ttl_hours=MATCHUP_CACHE_TTL_HOURS)

    logger.debug(
        "Matchup history batter %d vs. pitcher %d: PA=%d, xwOBA=%.3f",
        batter_id, pitcher_id, result["pa"], result.get("xwoba") or 0,
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_savant_csv(batter_id: int, pitcher_id: int) -> Optional[str]:
    """
    Hit Baseball Savant once; if HTTP 429, sleep 5 s and retry once.
    Returns raw CSV text string or None on failure.
    """
    url = SAVANT_CSV_URL.format(batter_id=batter_id, pitcher_id=pitcher_id)

    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 429:
                if attempt == 0:
                    logger.warning(
                        "Savant rate-limited (429) for batter %d vs pitcher %d. "
                        "Retrying after 5 s.",
                        batter_id, pitcher_id,
                    )
                    time.sleep(5)
                    continue
                else:
                    logger.warning(
                        "Savant still rate-limited on second attempt for batter %d "
                        "vs pitcher %d. Returning None.",
                        batter_id, pitcher_id,
                    )
                    return None
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as exc:
            logger.warning(
                "HTTP error fetching matchup CSV (batter %d vs pitcher %d): %s",
                batter_id, pitcher_id, exc,
            )
            return None
        except requests.RequestException as exc:
            logger.warning(
                "Network error fetching matchup CSV (batter %d vs pitcher %d): %s",
                batter_id, pitcher_id, exc,
            )
            return None

    return None


def _aggregate_matchup(df: pd.DataFrame) -> Optional[dict]:
    """
    Aggregate raw pitch-level data into career matchup summary.
    Returns None if PA < MIN_PA.
    """
    pa = len(df)
    if pa < MIN_PA:
        return None

    # AB: rows with at-bat ending events
    ab_ending = {
        "single", "double", "triple", "home_run", "strikeout",
        "field_out", "grounded_into_double_play", "double_play",
        "force_out", "fielders_choice", "fielders_choice_out",
        "sac_fly", "sac_bunt",
    }
    events_col = df.get("events", pd.Series(dtype=str)) if "events" in df.columns else pd.Series(dtype=str)
    ab = int(df["events"].isin(ab_ending).sum()) if "events" in df.columns else pa

    # Hits
    hit_events = {"single", "double", "triple", "home_run"}
    hits = int(df["events"].isin(hit_events).sum()) if "events" in df.columns else 0

    # HR
    hr = int((df["events"] == "home_run").sum()) if "events" in df.columns else 0

    # BB
    walk_events = {"walk", "intent_walk"}
    bb = int(df["events"].isin(walk_events).sum()) if "events" in df.columns else 0

    # K
    k = int((df["events"] == "strikeout").sum()) if "events" in df.columns else 0

    # xwOBA
    xwoba = None
    if "estimated_woba_using_speedangle" in df.columns:
        xwoba_vals = df["estimated_woba_using_speedangle"].dropna()
        xwoba = round(float(xwoba_vals.mean()), 4) if len(xwoba_vals) > 0 else None

    return {
        "pa": pa,
        "ab": ab,
        "hits": hits,
        "hr": hr,
        "bb": bb,
        "k": k,
        "xwoba": xwoba,
    }
