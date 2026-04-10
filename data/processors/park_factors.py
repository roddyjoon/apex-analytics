"""
Apex Analytics — Park Factor Processor
Fetches 3-year park factors from Baseball Savant's park factor leaderboard
and averages them for a stable run-scoring and HR-factor estimate per venue.

Data source: https://baseballsavant.mlb.com/leaderboard/statcast-park-factors
Response format: JSON with venue-level index_runs and index_hr fields.

Falls back to a hardcoded 2024-calibrated dict if the Savant endpoint fails.
"""

import json
import logging
from datetime import date
from typing import Optional

import requests

from config import PARK_FACTOR_YEARS, STADIUMS_FILE
from data.cache.db import save_park_factor, get_park_factor
from data.cache.file_cache import get_cache, set_cache, park_factor_cache_key

logger = logging.getLogger(__name__)

SAVANT_PARK_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=venue&year={year}&batSide=&stat=index_runs&roll=0&min=0"
)

# Cache: weekly refresh (park factors change slowly)
PARK_FACTOR_CACHE_TTL_HOURS = 7 * 24

# Hardcoded 2024 fallback: {team_abbr: (run_factor, hr_factor, venue_name)}
FALLBACK_PARK_FACTORS: dict[str, tuple[float, float, str]] = {
    "ARI": (1.02, 1.05, "Chase Field"),
    "ATL": (1.01, 1.02, "Truist Park"),
    "BAL": (1.03, 1.08, "Oriole Park at Camden Yards"),
    "BOS": (1.04, 0.98, "Fenway Park"),
    "CHC": (1.03, 1.04, "Wrigley Field"),
    "CWS": (0.96, 0.94, "Guaranteed Rate Field"),
    "CIN": (1.05, 1.10, "Great American Ball Park"),
    "CLE": (0.96, 0.93, "Progressive Field"),
    "COL": (1.15, 1.20, "Coors Field"),
    "DET": (0.97, 0.96, "Comerica Park"),
    "HOU": (0.99, 1.00, "Minute Maid Park"),
    "KC":  (0.98, 0.97, "Kauffman Stadium"),
    "LAA": (0.98, 0.97, "Angel Stadium"),
    "LAD": (0.96, 0.95, "Dodger Stadium"),
    "MIA": (0.95, 0.92, "loanDepot park"),
    "MIL": (0.98, 0.97, "American Family Field"),
    "MIN": (1.00, 1.01, "Target Field"),
    "NYM": (0.97, 0.96, "Citi Field"),
    "NYY": (1.03, 1.10, "Yankee Stadium"),
    "OAK": (0.94, 0.90, "Oakland Coliseum"),
    "PHI": (1.01, 1.02, "Citizens Bank Park"),
    "PIT": (0.97, 0.96, "PNC Park"),
    "SD":  (0.91, 0.87, "Petco Park"),
    "SEA": (0.95, 0.93, "T-Mobile Park"),
    "SF":  (0.91, 0.85, "Oracle Park"),
    "STL": (0.97, 0.95, "Busch Stadium"),
    "TB":  (0.97, 0.96, "Tropicana Field"),
    "TEX": (1.03, 1.05, "Globe Life Field"),
    "TOR": (1.00, 1.00, "Rogers Centre"),
    "WSH": (0.99, 0.98, "Nationals Park"),
    # Alias for Athletics relocation
    "ATH": (0.94, 0.90, "Oakland Coliseum"),
}

# Known mapping of Savant venue names → team abbreviations
# (Savant uses full venue names; we map them to team_abbr for downstream use)
VENUE_TO_ABBR: dict[str, str] = {
    "Chase Field":                    "ARI",
    "Truist Park":                    "ATL",
    "Oriole Park at Camden Yards":    "BAL",
    "Fenway Park":                    "BOS",
    "Wrigley Field":                  "CHC",
    "Guaranteed Rate Field":          "CWS",
    "Great American Ball Park":       "CIN",
    "Progressive Field":              "CLE",
    "Coors Field":                    "COL",
    "Comerica Park":                  "DET",
    "Minute Maid Park":               "HOU",
    "Kauffman Stadium":               "KC",
    "Angel Stadium":                  "LAA",
    "Dodger Stadium":                 "LAD",
    "loanDepot park":                 "MIA",
    "American Family Field":          "MIL",
    "Target Field":                   "MIN",
    "Citi Field":                     "NYM",
    "Yankee Stadium":                 "NYY",
    "Oakland Coliseum":               "OAK",
    "Citizens Bank Park":             "PHI",
    "PNC Park":                       "PIT",
    "Petco Park":                     "SD",
    "T-Mobile Park":                  "SEA",
    "Oracle Park":                    "SF",
    "Busch Stadium":                  "STL",
    "Tropicana Field":                "TB",
    "Globe Life Field":               "TEX",
    "Rogers Centre":                  "TOR",
    "Nationals Park":                 "WSH",
    # Alternate names that Savant has used historically
    "Globe Life Park in Arlington":   "TEX",
    "SunTrust Park":                  "ATL",
    "Oakland-Alameda County Coliseum":"OAK",
    "Marlins Park":                   "MIA",
    "Miller Park":                    "MIL",
    "Turner Field":                   "ATL",
}


def fetch_park_factors(force_refresh: bool = False) -> dict[str, dict]:
    """
    Fetch and average 3 years of park factors from Baseball Savant.

    Returns a dict keyed by team_abbr:
      {team_abbr: {run_factor, hr_factor, venue_name}}

    Falls back to FALLBACK_PARK_FACTORS if Savant is unreachable.
    Persists each venue to the DB via save_park_factor.
    Caches the full dict for PARK_FACTOR_CACHE_TTL_HOURS.
    """
    # Single cache key for the entire computed dict
    cache_key = "park_factors_all"

    if not force_refresh:
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

    current_year = date.today().year
    # Use prior full seasons (current season data is incomplete)
    target_years = [current_year - i for i in range(1, PARK_FACTOR_YEARS + 1)]

    season_dicts: list[dict[str, dict]] = []
    for year in target_years:
        season_data = _fetch_savant_year(year)
        if season_data:
            season_dicts.append(season_data)
            logger.debug("Fetched park factors for %d: %d venues.", year, len(season_data))
        else:
            logger.warning("Park factor fetch returned no data for %d.", year)

    if not season_dicts:
        logger.warning("All Savant park factor fetches failed. Using hardcoded fallback.")
        result = _build_fallback()
    else:
        result = _average_seasons(season_dicts)

    # Persist individual venue records to DB
    _persist_to_db(result)

    # Cache the full result
    set_cache(cache_key, result, ttl_hours=PARK_FACTOR_CACHE_TTL_HOURS)
    logger.info("Park factors computed for %d teams/venues.", len(result))
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_savant_year(year: int) -> dict[str, dict]:
    """
    Fetch one season's park factor JSON from Baseball Savant.
    Returns dict keyed by team_abbr, or empty dict on failure.
    """
    url = SAVANT_PARK_URL.format(year=year)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as exc:
        logger.warning("Savant park factor HTTP error for year %d: %s", year, exc)
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Savant park factor JSON parse error for year %d: %s", year, exc)
        return {}

    # The response is a list of venue records or wrapped in a key
    if isinstance(raw, dict):
        venue_list = raw.get("data", raw.get("venues", raw.get("results", [])))
    elif isinstance(raw, list):
        venue_list = raw
    else:
        logger.warning("Unexpected Savant park factor response format for year %d.", year)
        return {}

    if not venue_list:
        return {}

    result: dict[str, dict] = {}
    for venue in venue_list:
        venue_name = (
            venue.get("venue_name")
            or venue.get("name")
            or venue.get("Name", "")
        )
        team_abbr = _resolve_team_abbr(venue, venue_name)
        if not team_abbr:
            continue

        # index_runs: 100 = neutral; convert to factor (105 → 1.05)
        run_index = _safe_int(venue, ["index_runs", "run_index", "runs"], 100)
        hr_index = _safe_int(venue, ["index_hr", "hr_index", "hr"], 100)

        run_factor = round(run_index / 100.0, 4)
        hr_factor = round(hr_index / 100.0, 4)

        result[team_abbr] = {
            "run_factor": run_factor,
            "hr_factor": hr_factor,
            "venue_name": venue_name,
        }

    return result


def _resolve_team_abbr(venue: dict, venue_name: str) -> Optional[str]:
    """
    Attempt to determine team_abbr from a Savant venue record.
    Tries direct 'team_abbr' field, then VENUE_TO_ABBR mapping, then fuzzy prefix.
    """
    # Direct field
    if venue.get("team_abbr"):
        return str(venue["team_abbr"]).upper()
    if venue.get("abbreviation"):
        return str(venue["abbreviation"]).upper()

    # Known mapping
    if venue_name in VENUE_TO_ABBR:
        return VENUE_TO_ABBR[venue_name]

    # Fuzzy: check if venue_name contains any known key
    venue_lower = venue_name.lower()
    for known_name, abbr in VENUE_TO_ABBR.items():
        if known_name.lower() in venue_lower or venue_lower in known_name.lower():
            return abbr

    logger.debug("Could not resolve team_abbr for venue: '%s'", venue_name)
    return None


def _safe_int(venue: dict, keys: list[str], default: int) -> int:
    """Try multiple key names; return first found as int, else default."""
    for key in keys:
        val = venue.get(key)
        if val is not None:
            try:
                return int(float(str(val)))
            except (ValueError, TypeError):
                continue
    return default


def _average_seasons(season_dicts: list[dict[str, dict]]) -> dict[str, dict]:
    """
    Average run_factor and hr_factor across multiple season dicts.
    For teams missing from some seasons, use whatever seasons are present.
    Fill any teams not found in Savant data from FALLBACK_PARK_FACTORS.
    """
    aggregated: dict[str, list[dict]] = {}

    for season_data in season_dicts:
        for abbr, data in season_data.items():
            if abbr not in aggregated:
                aggregated[abbr] = []
            aggregated[abbr].append(data)

    result: dict[str, dict] = {}
    for abbr, records in aggregated.items():
        avg_run = round(sum(r["run_factor"] for r in records) / len(records), 4)
        avg_hr  = round(sum(r["hr_factor"]  for r in records) / len(records), 4)
        venue_name = records[-1]["venue_name"]  # use most recent season's name
        result[abbr] = {
            "run_factor": avg_run,
            "hr_factor": avg_hr,
            "venue_name": venue_name,
        }

    # Fill gaps from fallback
    for abbr, (run_f, hr_f, vname) in FALLBACK_PARK_FACTORS.items():
        if abbr not in result:
            result[abbr] = {
                "run_factor": run_f,
                "hr_factor": hr_f,
                "venue_name": vname,
            }

    return result


def _build_fallback() -> dict[str, dict]:
    """Convert FALLBACK_PARK_FACTORS to the standard output format."""
    return {
        abbr: {"run_factor": run_f, "hr_factor": hr_f, "venue_name": vname}
        for abbr, (run_f, hr_f, vname) in FALLBACK_PARK_FACTORS.items()
    }


def _persist_to_db(park_factors: dict[str, dict]) -> None:
    """
    Persist each venue's computed factors to the DB.
    Uses a synthetic venue_id derived from the team_abbr for now;
    the DB ParkFactor table also stores venue_id.
    """
    # Load stadiums for venue_id lookup
    stadiums = _load_stadiums()
    current_year = date.today().year

    for abbr, data in park_factors.items():
        venue_id = stadiums.get(abbr, {}).get("venue_id", 0)
        try:
            save_park_factor(venue_id, {
                "venue_name": data["venue_name"],
                "team_abbr": abbr,
                "season_through": current_year - 1,
                "years_averaged": PARK_FACTOR_YEARS,
                "run_factor": data["run_factor"],
                "hr_factor": data["hr_factor"],
            })
        except Exception as exc:
            logger.debug("Failed to persist park factor for %s: %s", abbr, exc)


def get_park_factor_for_team(
    team_abbr: str, park_factors: Optional[dict] = None
) -> dict:
    """
    Look up a single team's park factor from an already-fetched dict,
    or fetch all factors and return the matching team.
    Returns neutral {run_factor: 1.0, hr_factor: 1.0} if not found.
    """
    if park_factors is None:
        park_factors = fetch_park_factors()

    return park_factors.get(
        team_abbr,
        {"run_factor": 1.0, "hr_factor": 1.0, "venue_name": "Unknown"},
    )


# ---------------------------------------------------------------------------
# Stadium helper
# ---------------------------------------------------------------------------

_stadiums_loaded: Optional[dict] = None


def _load_stadiums() -> dict:
    """Load stadiums.json and return dict keyed by team_abbr."""
    global _stadiums_loaded
    if _stadiums_loaded is not None:
        return _stadiums_loaded
    try:
        with open(STADIUMS_FILE, "r") as f:
            raw = json.load(f)
        _stadiums_loaded = {s["team_abbr"]: s for s in raw.get("stadiums", [])}
    except Exception as exc:
        logger.warning("Could not load stadiums.json for park factor DB persist: %s", exc)
        _stadiums_loaded = {}
    return _stadiums_loaded
