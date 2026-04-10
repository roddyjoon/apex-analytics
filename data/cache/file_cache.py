"""
Apex Analytics — File Cache
JSON-on-disk cache with TTL for API responses and computed data.
Falls back gracefully if the cache directory is unwritable.

Cache directory: $APEX_CACHE_DIR (default: /tmp/apex_file_cache)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
STATCAST_CACHE_TTL_HOURS = 6.0   # Statcast data: refresh every 6 hours
_CACHE_DIR = Path(os.environ.get("APEX_CACHE_DIR", "/tmp/apex_file_cache"))

try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    _CACHE_DIR = Path("/tmp/apex_file_cache_fallback")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Core get/set ───────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    """Return the filesystem path for a given cache key."""
    safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
    return _CACHE_DIR / f"{safe}.json"


def get_cache(key: str) -> Optional[Any]:
    """
    Return cached value for `key`, or None if missing / expired.
    """
    path = _cache_path(key)
    try:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        entry = json.loads(raw)
        expires_at = entry.get("expires_at", 0)
        if expires_at and time.time() > expires_at:
            path.unlink(missing_ok=True)
            return None
        return entry.get("value")
    except Exception as exc:
        logger.debug("Cache read error for key=%s: %s", key, exc)
        return None


def set_cache(key: str, value: Any, ttl_hours: float = 24.0) -> None:
    """
    Persist `value` to cache with a TTL.
    """
    path = _cache_path(key)
    try:
        entry = {
            "value":      value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl_hours * 3600,
        }
        path.write_text(json.dumps(entry, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("Cache write error for key=%s: %s", key, exc)


def invalidate_cache(key: str) -> None:
    """Remove a single cache entry."""
    path = _cache_path(key)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ── Cache key helpers ──────────────────────────────────────────────────────────

def schedule_cache_key(date_str: str) -> str:
    """e.g. 'schedule_2026-04-10'"""
    return f"schedule_{date_str}"


def lineup_cache_key(game_pk: int, report_type: str) -> str:
    """e.g. 'lineup_745528_morning'"""
    return f"lineup_{game_pk}_{report_type}"


def weather_cache_key(lat: float, lon: float, date_str: str) -> str:
    """e.g. 'weather_40.71_-74.0_2026-04-10'"""
    return f"weather_{lat:.4f}_{lon:.4f}_{date_str}"


def statcast_cache_key(data_type: str, player_id: int, season: int) -> str:
    """e.g. 'statcast_batter_123456_2026'"""
    return f"statcast_{data_type}_{player_id}_{season}"


def park_factor_cache_key(team_abbr: str) -> str:
    """e.g. 'park_factor_NYY'"""
    return f"park_factor_{team_abbr.upper()}"


def matchup_cache_key(batter_id: int, pitcher_id: int) -> str:
    """e.g. 'matchup_111_222'"""
    return f"matchup_{batter_id}_{pitcher_id}"


def pitch_splits_cache_key(player_id: int, season: int) -> str:
    """e.g. 'pitch_splits_123456_2026'"""
    return f"pitch_splits_{player_id}_{season}"


def arsenal_cache_key(pitcher_id: int, season: int) -> str:
    """e.g. 'arsenal_123456_2026'"""
    return f"arsenal_{pitcher_id}_{season}"
