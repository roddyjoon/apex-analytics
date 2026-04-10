"""
Apex Analytics — Weather Ingestor
Fetches forecast weather from Open-Meteo (free, no API key required) and
translates conditions into run-scoring adjustments used by the Monte Carlo engine.

Wind direction math:
  angle_diff = abs((wind_from_deg - cf_orientation_deg + 180) % 360 - 180)
  < 45  → blowing OUT (wind from behind home plate toward CF)
  > 135 → blowing IN  (wind from CF toward home plate)
  else  → crosswind
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from config import (
    WIND_OUT_BRACKETS,
    WIND_IN_BRACKETS,
    WIND_CROSS_ADJ,
    WIND_CALM_THRESHOLD,
    TEMP_ADJUSTMENTS,
    STADIUMS_FILE,
    STATCAST_CACHE_TTL_HOURS,
)
from data.cache.file_cache import get_cache, set_cache, weather_cache_key

logger = logging.getLogger(__name__)

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m"
    "&temperature_unit=fahrenheit"
    "&windspeed_unit=mph"
    "&forecast_days=1"
    "&timezone=auto"
)

# Weather cache is short: 2 hours so pre-game report picks up updated forecasts
WEATHER_CACHE_TTL_HOURS = 2.0


def fetch_weather(
    lat: float,
    lon: float,
    game_date: date,
    game_time_utc: Optional[datetime] = None,
    is_dome: bool = False,
    cf_orientation_deg: Optional[float] = None,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch weather for a stadium and compute run-scoring adjustments.

    Parameters
    ----------
    lat, lon          : Stadium coordinates.
    game_date         : Date of the game.
    game_time_utc     : Game start time in UTC (optional). Defaults to 19:00 local if None.
    is_dome           : If True, skip fetch and return zero adjustments.
    cf_orientation_deg: Stadium CF orientation in degrees (from stadiums.json).
    force_refresh     : Bypass file cache.

    Returns
    -------
    dict with keys:
      temp_f, wind_speed_mph, wind_direction_deg, wind_classification,
      wind_run_adj, temp_run_adj, net_run_adj, humidity_pct,
      is_dome, weather_note
    """
    if is_dome:
        return _dome_result()

    date_str = game_date.isoformat()
    cache_key = weather_cache_key(lat, lon, date_str)

    if not force_refresh:
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

    url = OPEN_METEO_URL.format(lat=lat, lon=lon)

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("Open-Meteo fetch failed (lat=%.4f, lon=%.4f): %s", lat, lon, exc)
        return _fallback_weather()

    try:
        result = _parse_weather(
            data,
            game_time_utc=game_time_utc,
            cf_orientation_deg=cf_orientation_deg,
            is_dome=is_dome,
        )
    except Exception as exc:
        logger.error("Weather parse failed: %s", exc)
        return _fallback_weather()

    set_cache(cache_key, result, ttl_hours=WEATHER_CACHE_TTL_HOURS)
    logger.debug(
        "Weather fetched (lat=%.4f, lon=%.4f): %s",
        lat, lon, result.get("weather_note", ""),
    )
    return result


def fetch_weather_for_game(game: dict, force_refresh: bool = False) -> dict:
    """
    High-level convenience: look up stadium data from stadiums.json using the
    game's home team abbreviation, then call fetch_weather.

    Parameters
    ----------
    game : dict from fetch_schedule(), must have 'home_team_abbr' and 'game_time_utc'.
    """
    stadiums = _load_stadiums()
    home_abbr = game.get("home_team_abbr", "")
    stadium = stadiums.get(home_abbr)

    if stadium is None:
        logger.warning(
            "No stadium data for team '%s'. Using fallback weather.", home_abbr
        )
        return _fallback_weather()

    return fetch_weather(
        lat=stadium["lat"],
        lon=stadium["lon"],
        game_date=game.get("game_date", date.today()),
        game_time_utc=game.get("game_time_utc"),
        is_dome=stadium.get("dome", False),
        cf_orientation_deg=stadium.get("cf_orientation_deg"),
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_weather(
    data: dict,
    game_time_utc: Optional[datetime],
    cf_orientation_deg: Optional[float],
    is_dome: bool,
) -> dict:
    """Parse Open-Meteo JSON and compute run adjustments."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    wind_speeds = hourly.get("windspeed_10m", [])
    wind_dirs = hourly.get("winddirection_10m", [])
    humidities = hourly.get("relativehumidity_2m", [])

    if not times:
        return _fallback_weather()

    # Find index closest to game time
    target_idx = _find_closest_hour_index(times, game_time_utc)

    temp_f = _safe_float(temps, target_idx, 72.0)
    wind_speed_mph = _safe_float(wind_speeds, target_idx, 0.0)
    wind_dir_deg = _safe_float(wind_dirs, target_idx, 0.0)
    humidity_pct = _safe_float(humidities, target_idx, 50.0)

    # Wind classification
    wind_class, wind_run_adj = _classify_wind(
        wind_speed_mph, wind_dir_deg, cf_orientation_deg
    )

    # Temperature adjustment
    temp_run_adj = _temp_adjustment(temp_f)

    net_run_adj = round(wind_run_adj + temp_run_adj, 2)

    weather_note = _build_note(
        temp_f, wind_speed_mph, wind_dir_deg, wind_class, wind_run_adj, temp_run_adj
    )

    return {
        "temp_f": round(temp_f, 1),
        "wind_speed_mph": round(wind_speed_mph, 1),
        "wind_direction_deg": round(wind_dir_deg, 1),
        "wind_classification": wind_class,
        "wind_run_adj": round(wind_run_adj, 2),
        "temp_run_adj": round(temp_run_adj, 2),
        "net_run_adj": net_run_adj,
        "humidity_pct": round(humidity_pct, 1),
        "is_dome": is_dome,
        "weather_note": weather_note,
    }


def _find_closest_hour_index(times: list, game_time_utc: Optional[datetime]) -> int:
    """
    Find the index in the hourly times list closest to the game start time.
    If game_time_utc is None, default to the 19:00 entry (7 PM local, index 19).
    """
    if game_time_utc is None:
        # Default: 19:00 local — index 19 in a full 24-hour hourly array
        return min(19, len(times) - 1)

    # Convert game_time_utc to naive UTC for string comparison
    if game_time_utc.tzinfo is not None:
        game_dt_utc = game_time_utc.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        game_dt_utc = game_time_utc

    best_idx = 0
    best_diff = float("inf")
    for i, t_str in enumerate(times):
        try:
            t_dt = datetime.fromisoformat(t_str)
        except ValueError:
            continue
        diff = abs((t_dt - game_dt_utc).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return best_idx


def _classify_wind(
    wind_speed_mph: float,
    wind_from_deg: float,
    cf_orientation_deg: Optional[float],
) -> tuple[str, float]:
    """
    Return (classification, run_adjustment) for the given wind.
    classification: 'calm', 'out', 'in', 'cross'
    """
    if wind_speed_mph < WIND_CALM_THRESHOLD:
        return "calm", 0.0

    if cf_orientation_deg is None:
        # No orientation data — treat as crosswind
        return "cross", _bracket_adj(wind_speed_mph, {}) or WIND_CROSS_ADJ

    # Angle between wind-from direction and CF orientation
    angle_diff = abs((wind_from_deg - cf_orientation_deg + 180) % 360 - 180)

    if angle_diff < 45:
        classification = "out"
        adj = _bracket_adj(wind_speed_mph, WIND_OUT_BRACKETS)
    elif angle_diff > 135:
        classification = "in"
        adj = _bracket_adj(wind_speed_mph, WIND_IN_BRACKETS)
    else:
        classification = "cross"
        adj = WIND_CROSS_ADJ

    return classification, adj


def _bracket_adj(wind_speed_mph: float, brackets: dict) -> float:
    """Look up run adjustment from a speed-bracket dict."""
    for (low, high), adj in brackets.items():
        if low <= wind_speed_mph <= high:
            return adj
    return 0.0


def _temp_adjustment(temp_f: float) -> float:
    """Look up run adjustment from the TEMP_ADJUSTMENTS config bracket dict."""
    for (low, high), adj in TEMP_ADJUSTMENTS.items():
        if low <= temp_f <= high:
            return adj
    return 0.0


def _compute_wind_adj(wind_speed_mph: float, wind_classification: str) -> float:
    """
    Return run adjustment for the given wind speed and classification.
    Convenience wrapper around _bracket_adj for external callers and tests.

    wind_classification: one of "OUT", "IN", "CALM", "CROSS" (case-insensitive)
    """
    cls = wind_classification.upper()
    if cls == "OUT":
        return _bracket_adj(wind_speed_mph, WIND_OUT_BRACKETS)
    if cls == "IN":
        return _bracket_adj(wind_speed_mph, WIND_IN_BRACKETS)
    return 0.0  # CALM or CROSS → no significant run adjustment


def _compute_temp_adj(temp_f: float) -> float:
    """
    Return run adjustment for the given temperature.
    Convenience wrapper around _temp_adjustment for external callers and tests.
    """
    return _temp_adjustment(temp_f)


def _safe_float(lst: list, idx: int, default: float) -> float:
    """Safely extract a float from a list at the given index."""
    try:
        val = lst[idx]
        return float(val) if val is not None else default
    except (IndexError, TypeError, ValueError):
        return default


def _build_note(
    temp_f: float,
    wind_speed_mph: float,
    wind_dir_deg: float,
    wind_class: str,
    wind_run_adj: float,
    temp_run_adj: float,
) -> str:
    """Build a human-readable weather summary string."""
    parts = [f"{temp_f:.0f}°F"]

    if wind_speed_mph >= WIND_CALM_THRESHOLD:
        sign = "+" if wind_run_adj >= 0 else ""
        parts.append(
            f"{wind_speed_mph:.0f} mph wind {wind_class} "
            f"({sign}{wind_run_adj:.1f} R/G)"
        )
    else:
        parts.append("calm wind")

    if temp_run_adj != 0:
        sign = "+" if temp_run_adj >= 0 else ""
        parts.append(f"temp adj {sign}{temp_run_adj:.1f} R/G")

    return ", ".join(parts)


def _apply_dome_override(is_dome: bool, wind_adj: float, temp_adj: float) -> float:
    """
    Return net run adjustment, zeroing everything out for dome stadiums.
    Domes eliminate all weather impact — temperature and wind are controlled.

    Args:
        is_dome:  True if the venue is an enclosed dome
        wind_adj: Wind run adjustment (positive = hitter-friendly)
        temp_adj: Temperature run adjustment
    Returns:
        Net run adjustment (0.0 for domes, wind_adj + temp_adj otherwise)
    """
    if is_dome:
        return 0.0
    return wind_adj + temp_adj


def _dome_result() -> dict:
    """Return a neutral weather dict for domed stadiums."""
    return {
        "temp_f": 72.0,
        "wind_speed_mph": 0.0,
        "wind_direction_deg": 0.0,
        "wind_classification": "calm",
        "wind_run_adj": 0.0,
        "temp_run_adj": 0.0,
        "net_run_adj": 0.0,
        "humidity_pct": 50.0,
        "is_dome": True,
        "weather_note": "dome",
    }


def _fallback_weather() -> dict:
    """Return neutral weather when no data is available."""
    return {
        "temp_f": 72.0,
        "wind_speed_mph": 0.0,
        "wind_direction_deg": 0.0,
        "wind_classification": "calm",
        "wind_run_adj": 0.0,
        "temp_run_adj": 0.0,
        "net_run_adj": 0.0,
        "humidity_pct": 50.0,
        "is_dome": False,
        "weather_note": "weather unavailable — using neutral defaults",
    }


# ---------------------------------------------------------------------------
# Stadium lookup helpers
# ---------------------------------------------------------------------------

_stadiums_cache: Optional[dict] = None


def _load_stadiums() -> dict:
    """Load stadiums.json and return dict keyed by team_abbr."""
    global _stadiums_cache
    if _stadiums_cache is not None:
        return _stadiums_cache

    try:
        with open(STADIUMS_FILE, "r") as f:
            raw = json.load(f)
        _stadiums_cache = {s["team_abbr"]: s for s in raw.get("stadiums", [])}
    except Exception as exc:
        logger.error("Failed to load stadiums.json: %s", exc)
        _stadiums_cache = {}

    return _stadiums_cache
