"""
Apex Analytics — Profile Builder (Orchestrator)
Assembles BatterProfile, PitcherProfile, and GameContext objects
from raw Statcast data with all enhancements applied in sequence:

  1. Fetch current-season Statcast stats
  2. Apply Bayesian prior blend (prior season actuals)
  3. Apply exponential decay recency weighting
  4. Apply home/away ERA split (pitchers)
  5. Compute true SIERA + FIP + true-talent ERA blend (pitchers)
  6. Attach pitch-type vulnerability splits (batters)
  7. Attach pitcher arsenal (pitchers)
  8. Attach batter-vs-pitcher matchup history
  9. Set confidence flags
  10. Return simulation-ready profile objects
"""

import logging
from datetime import date
from typing import Optional

from config import (
    FIP_CONSTANT,
    ERA_BLEND_SIERA,
    ERA_BLEND_XERA,
    ERA_BLEND_FIP,
    LEAGUE_AVG_XWOBA,
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_SPRINT_SPEED,
    LEAGUE_AVG_SWSTR_PCT,
    LOW_CONFIDENCE_PA_THRESH,
    LOW_CONFIDENCE_BF_THRESH,
)
from data.cache.db import get_pitch_type_splits, get_pitcher_arsenal as db_get_arsenal
from data.ingestors.statcast_batter import fetch_batter_stats, fetch_batter_platoon_splits
from data.ingestors.statcast_pitcher import fetch_pitcher_stats
from data.ingestors.pitch_arsenal import fetch_pitch_arsenal
from data.ingestors.batter_pitch_splits import fetch_batter_pitch_splits
from data.ingestors.matchup_history import fetch_matchup_history
from data.processors.siera_calculator import (
    compute_siera, compute_fip, blend_true_talent_era, compute_stuff_plus_proxy
)
from data.processors.recency_weighter import (
    compute_batter_recent_xwoba, compute_pitcher_recent_xera,
    compute_pitcher_recent_k_pct, compute_pitcher_recent_bb_pct,
)
from data.processors.bayesian_prior import blend_batter_stats, blend_pitcher_stats
from simulation.profiles import BatterProfile, PitcherProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batter Profile Builder
# ---------------------------------------------------------------------------


def build_batter_profile(
    player_id:    int,
    player_name:  str,
    batting_order: int,
    position:     str,
    bats:         str,
    opposing_pitcher_id: Optional[int] = None,
    is_confirmed: bool = False,
    is_projected: bool = True,
    season:       int = 2025,
    game_date:    Optional[date] = None,
) -> BatterProfile:
    """
    Build a fully-enriched BatterProfile for use in Monte Carlo simulation.

    Parameters
    ----------
    player_id           : MLB Stats API person ID.
    player_name         : Full player name for display.
    batting_order       : Position in lineup (1-9).
    position            : Fielding position abbreviation.
    bats                : "L", "R", or "S".
    opposing_pitcher_id : Pitcher they'll face — used for matchup history lookup.
    is_confirmed        : Is this from a confirmed lineup?
    is_projected        : True if lineup is projected (not confirmed).
    season              : Current season year.
    game_date           : Game date for context (defaults to today).
    """
    if game_date is None:
        game_date = date.today()

    # ── Step 1: Fetch current-season Statcast stats ──────────────────────────
    raw_stats = fetch_batter_stats(player_id, season)
    if raw_stats is None:
        raw_stats = {}

    # ── Step 2: Bayesian prior blend ─────────────────────────────────────────
    blended = blend_batter_stats(raw_stats, player_id, season)

    # ── Step 3: Exponential decay recency weighting ───────────────────────────
    game_log = raw_stats.get("game_log", [])
    exp_decay_xwoba = compute_batter_recent_xwoba(game_log, reference_date=game_date)

    # ── Step 4: Platoon splits ────────────────────────────────────────────────
    platoon = fetch_batter_platoon_splits(player_id, season)

    # Blend platoon splits with overall using Bayesian weight
    pa = blended.get("pa", 0)
    in_wt = min(1.0, pa / 600)
    xwoba_vs_lhp = (
        platoon.get("xwoba_vs_lhp", blended["xwoba"]) * in_wt
        + blended["xwoba"] * (1 - in_wt)
    )
    xwoba_vs_rhp = (
        platoon.get("xwoba_vs_rhp", blended["xwoba"]) * in_wt
        + blended["xwoba"] * (1 - in_wt)
    )

    # ── Step 5: Pitch-type splits ─────────────────────────────────────────────
    pitch_splits_list = get_pitch_type_splits(player_id, season)
    if not pitch_splits_list:
        # Try fetching fresh
        pitch_splits_list = fetch_batter_pitch_splits(player_id, season)

    # Build {pitch_type: {xwoba, pa, k_pct}} dict
    pitch_type_xwoba = {}
    for split in pitch_splits_list:
        pt = split.get("pitch_type")
        if pt:
            pitch_type_xwoba[pt] = {
                "xwoba": split.get("xwoba", LEAGUE_AVG_XWOBA),
                "pa":    split.get("pa", 0),
                "k_pct": split.get("k_pct", 0.228),
            }

    # ── Step 6: Matchup history ───────────────────────────────────────────────
    matchup_dict = {}
    if opposing_pitcher_id is not None:
        history = fetch_matchup_history(player_id, opposing_pitcher_id)
        if history and history.get("pa", 0) >= 10:
            matchup_dict[opposing_pitcher_id] = {
                "pa":    history["pa"],
                "xwoba": history.get("xwoba", blended["xwoba"]),
                "hr":    history.get("hr", 0),
                "k":     history.get("k", 0),
                "bb":    history.get("bb", 0),
            }

    # ── Assemble profile ──────────────────────────────────────────────────────
    profile = BatterProfile(
        player_id=player_id,
        player_name=player_name,
        batting_order=batting_order,
        position=position,
        bats=bats,

        xwoba=blended.get("xwoba", LEAGUE_AVG_XWOBA),
        xba=blended.get("xba", 0.250),
        barrel_pct=blended.get("barrel_pct", 0.080),
        hard_hit_pct=blended.get("hard_hit_pct", 0.380),
        swstr_pct=blended.get("swstr_pct", LEAGUE_AVG_SWSTR_PCT),
        k_pct=blended.get("k_pct", 0.228),
        bb_pct=blended.get("bb_pct", 0.083),
        hr_rate=blended.get("hr_rate", 0.037),
        obp=blended.get("obp", 0.315),
        slg=blended.get("slg", 0.413),
        sprint_speed=blended.get("sprint_speed", LEAGUE_AVG_SPRINT_SPEED),

        xwoba_vs_lhp=xwoba_vs_lhp,
        xwoba_vs_rhp=xwoba_vs_rhp,
        pa_vs_lhp=platoon.get("pa_vs_lhp", 0),
        pa_vs_rhp=platoon.get("pa_vs_rhp", 0),

        exp_decay_xwoba=exp_decay_xwoba,
        pitch_type_xwoba=pitch_type_xwoba,
        matchup_history=matchup_dict,

        pa=pa,
        pa_season=pa,
        confidence=blended.get("confidence", "LOW"),
        is_confirmed=is_confirmed,
        is_projected=is_projected,
    )

    logger.debug(
        "Built BatterProfile: %s (id=%d), xwOBA=%.3f, decay=%.3f, PA=%d, conf=%s",
        player_name, player_id,
        profile.xwoba,
        exp_decay_xwoba or 0,
        pa,
        profile.confidence,
    )
    return profile


# ---------------------------------------------------------------------------
# Pitcher Profile Builder
# ---------------------------------------------------------------------------


def build_pitcher_profile(
    player_id:   int,
    player_name: str,
    throws:      str,
    team_id:     int,
    is_home:     bool,
    is_starter:  bool = True,
    is_opener:   bool = False,
    season:      int = 2025,
    game_date:   Optional[date] = None,
) -> PitcherProfile:
    """
    Build a fully-enriched PitcherProfile for use in the simulation engine.
    """
    if game_date is None:
        game_date = date.today()

    # ── Step 1: Fetch Statcast pitcher stats ──────────────────────────────────
    raw_stats = fetch_pitcher_stats(player_id, season, team_id=team_id)
    if raw_stats is None:
        raw_stats = {}

    # ── Step 2: Bayesian prior blend ─────────────────────────────────────────
    blended = blend_pitcher_stats(raw_stats, player_id, season)

    # ── Step 3: Compute true SIERA ────────────────────────────────────────────
    siera = compute_siera(
        k_pct=blended.get("k_pct", 0.228),
        bb_pct=blended.get("bb_pct", 0.083),
        gb_pct=blended.get("gb_pct", 0.430),
        fb_pct=blended.get("fb_pct", 0.350),
        ld_pct=blended.get("ld_pct", 0.220),
    )

    # ── Step 4: Compute FIP ───────────────────────────────────────────────────
    fip = compute_fip(
        k_pct=blended.get("fip_k_pct", blended.get("k_pct", 0.228)),
        bb_pct=blended.get("fip_bb_pct", blended.get("bb_pct", 0.083)),
        hr_rate=blended.get("fip_hr_rate", 0.037),
        fip_constant=FIP_CONSTANT,
    )

    # ── Step 5: True-talent ERA blend ─────────────────────────────────────────
    xera = blended.get("xera", LEAGUE_AVG_ERA)
    true_talent_era = blend_true_talent_era(
        siera=siera,
        xera=xera,
        fip=fip,
        siera_weight=ERA_BLEND_SIERA,
        xera_weight=ERA_BLEND_XERA,
        fip_weight=ERA_BLEND_FIP,
    )

    # ── Step 6: Home/away ERA adjustment ─────────────────────────────────────
    home_adj = raw_stats.get("home_era_adj", 1.0)
    away_adj = raw_stats.get("away_era_adj", 1.0)
    location_adj = home_adj if is_home else away_adj
    adjusted_era = true_talent_era * location_adj

    # ── Step 7: Exponential decay (last 5 starts) ─────────────────────────────
    game_log = raw_stats.get("game_log", [])
    exp_decay_xera   = compute_pitcher_recent_xera(game_log, reference_date=game_date)
    exp_decay_k_pct  = compute_pitcher_recent_k_pct(game_log, reference_date=game_date)
    exp_decay_bb_pct = compute_pitcher_recent_bb_pct(game_log, reference_date=game_date)

    # ── Step 8: Pitch arsenal ─────────────────────────────────────────────────
    arsenal_list = db_get_arsenal(player_id, season)
    if not arsenal_list:
        arsenal_list = fetch_pitch_arsenal(player_id, season)

    arsenal_dict = {
        entry["pitch_type"]: {
            "usage_pct":      entry.get("usage_pct", 0.0),
            "xwoba_conceded": entry.get("xwoba_conceded", LEAGUE_AVG_XWOBA),
            "swstr_pct":      entry.get("swstr_pct", LEAGUE_AVG_SWSTR_PCT),
            "avg_velocity":   entry.get("avg_velocity"),
        }
        for entry in arsenal_list
        if entry.get("pitch_type")
    }

    # ── Step 9: Stuff+ proxy ──────────────────────────────────────────────────
    stuff_plus = compute_stuff_plus_proxy(
        swstr_pct=blended.get("swstr_pct", LEAGUE_AVG_SWSTR_PCT)
    )

    # ── Step 10: Days rest ────────────────────────────────────────────────────
    days_rest = raw_stats.get("days_rest", 4)

    # ── Assemble profile ──────────────────────────────────────────────────────
    bf = blended.get("bf", 0)

    profile = PitcherProfile(
        player_id=player_id,
        player_name=player_name,
        throws=throws,
        is_starter=is_starter,
        is_opener=is_opener,

        true_talent_era=round(adjusted_era, 3),
        xera=round(xera, 3),
        fip=round(fip, 3),
        siera=round(siera, 3),

        k_pct=blended.get("k_pct", 0.228),
        bb_pct=blended.get("bb_pct", 0.083),
        gb_pct=blended.get("gb_pct", 0.430),
        fb_pct=blended.get("fb_pct", 0.350),
        ld_pct=blended.get("ld_pct", 0.220),

        swstr_pct=blended.get("swstr_pct", LEAGUE_AVG_SWSTR_PCT),
        csw_pct=blended.get("csw_pct", 0.280),
        barrel_pct_allowed=blended.get("barrel_pct_allowed", 0.080),
        xba_allowed=blended.get("xba_allowed", 0.250),

        home_era_adj=home_adj,
        away_era_adj=away_adj,

        exp_decay_xera=exp_decay_xera,
        exp_decay_k_pct=exp_decay_k_pct,
        exp_decay_bb_pct=exp_decay_bb_pct,

        arsenal=arsenal_dict,
        fatigue_index=0.0,
        days_rest=days_rest,
        pitch_count_est=0,
        stuff_plus_proxy=stuff_plus,

        bf=bf,
        confidence=blended.get("confidence", "LOW"),
        is_on_il=False,
        is_home=is_home,
    )

    logger.debug(
        "Built PitcherProfile: %s (id=%d), TT_ERA=%.2f [SIERA=%.2f, xERA=%.2f, FIP=%.2f], "
        "BF=%d, conf=%s, arsenal=%s",
        player_name, player_id,
        profile.true_talent_era, profile.siera, profile.xera, profile.fip,
        bf, profile.confidence,
        list(arsenal_dict.keys()),
    )
    return profile


# ---------------------------------------------------------------------------
# Lineup-level builders
# ---------------------------------------------------------------------------


def build_lineup_profiles(
    lineup_slots: list[dict],
    opposing_pitcher_id: Optional[int],
    season: int = 2025,
    game_date: Optional[date] = None,
    is_confirmed: bool = False,
) -> list[BatterProfile]:
    """
    Build a list of BatterProfiles from raw lineup slot dicts.

    lineup_slots: List of {"batting_order", "player_id", "player_name",
                           "position", "is_confirmed", "source"} dicts.
    """
    profiles = []
    for slot in lineup_slots:
        pid = slot.get("player_id")
        if pid is None:
            continue

        # Fetch bats from DB player record
        from data.cache.db import get_player
        player_rec = get_player(pid)
        bats = player_rec.get("bats", "R") if player_rec else "R"

        profile = build_batter_profile(
            player_id=pid,
            player_name=slot.get("player_name", f"Player {pid}"),
            batting_order=slot.get("batting_order", 9),
            position=slot.get("position", ""),
            bats=bats,
            opposing_pitcher_id=opposing_pitcher_id,
            is_confirmed=slot.get("is_confirmed", False),
            is_projected=not slot.get("is_confirmed", False),
            season=season,
            game_date=game_date,
        )
        profiles.append(profile)

    # Sort by batting order
    profiles.sort(key=lambda p: p.batting_order)
    return profiles


# ---------------------------------------------------------------------------
# Fallback / TBD Pitcher Profile
# ---------------------------------------------------------------------------

def _build_fallback_pitcher_profile(
    player_id:   int,
    player_name: str,
    throws:      str = "R",
    is_home:     bool = True,
) -> PitcherProfile:
    """
    Return a replacement-level PitcherProfile when no Statcast data is available.
    Used for TBD starters, missing pitcher data, or roster-fallback scenarios.
    The profile uses conservative (pessimistic) replacement-level stats and
    is tagged with confidence='LOW' to flag uncertainty in the report.
    """
    era = 5.50  # Replacement-level ERA (league average ~4.20; replacement ~5.50)

    return PitcherProfile(
        player_id            = player_id,
        player_name          = player_name,
        throws               = throws,
        is_starter           = True,
        is_opener            = False,
        true_talent_era      = era,
        xera                 = era,
        fip                  = era + 0.10,
        siera                = era + 0.05,
        k_pct                = 0.18,
        bb_pct               = 0.10,
        gb_pct               = 0.42,
        fb_pct               = 0.38,
        ld_pct               = 0.20,
        swstr_pct            = LEAGUE_AVG_SWSTR_PCT,
        csw_pct              = 0.26,
        barrel_pct_allowed   = 0.10,
        xba_allowed          = 0.270,
        home_era_adj         = 1.0,
        away_era_adj         = 1.0,
        exp_decay_xera       = era,
        exp_decay_k_pct      = 0.18,
        exp_decay_bb_pct     = 0.10,
        arsenal              = {"FF": 0.55, "SL": 0.25, "CH": 0.15, "CU": 0.05},
        fatigue_index        = 0.0,
        days_rest            = 4,
        pitch_count_est      = 0,
        stuff_plus_proxy     = 85.0,
        bf                   = 0,
        confidence           = "LOW",
        is_on_il             = False,
        is_home              = is_home,
    )
