"""
Apex Analytics — Report Explainer
Generates human-readable "Key Factors" bullets and weather/lineup summaries.

The explainer answers: "Why does the model lean home 59%?"
It ranks contributing factors by magnitude of win probability impact
and formats them as plain-English bullets for the report.

Factor types analyzed:
  - SP ERA differential (true-talent ERA)
  - SP K-rate differential (SwStr%, CSW%)
  - Lineup xwOBA differential (avg across 9 hitters)
  - Bullpen xFIP differential
  - Decay win% differential (recent form)
  - Weather impact (net run adjustment)
  - Park factor impact vs neutral baseline
  - Bullpen fatigue flag
  - Elo differential
  - Home field advantage
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Impact thresholds for factor inclusion (minimum magnitude to show)
ERA_THRESHOLD      = 0.25    # ERA difference worth mentioning
XWOBA_THRESHOLD    = 0.010   # xwOBA difference worth mentioning (10 pts)
BULLPEN_THRESHOLD  = 0.20    # Bullpen xFIP difference
WIN_PCT_THRESHOLD  = 0.040   # Decay win% difference (4 percentage points)
WEATHER_THRESHOLD  = 0.20    # Net run adjustment worth mentioning
PARK_THRESHOLD     = 0.03    # Park factor deviation from 1.00
ELO_THRESHOLD      = 30.0    # Elo point difference worth mentioning


def explain_game(
    game_context,
    mc_result:       dict,
    ensemble_result: dict,
    max_factors:     int = 7,
) -> list:
    """
    Generate ranked plain-English factor bullets explaining the win probability.

    Parameters
    ----------
    game_context     : GameContext object with both teams' profiles.
    mc_result        : Output from monte_carlo.run_monte_carlo().
    ensemble_result  : Output from blender.blend_predictions().
    max_factors      : Maximum number of factors to return (default 7).

    Returns
    -------
    list[str] — Top factors ranked by magnitude, formatted as display strings.
    """
    factors = []  # List of (magnitude, display_str) tuples

    home_sp  = game_context.home_starter
    away_sp  = game_context.away_starter
    home_bp  = game_context.home_bullpen
    away_bp  = game_context.away_bullpen
    park     = game_context.park

    # ── 1. SP ERA differential ────────────────────────────────────────────────
    if home_sp and away_sp:
        home_era = getattr(home_sp, "true_talent_era", 4.20)
        away_era = getattr(away_sp, "true_talent_era", 4.20)
        diff = abs(home_era - away_era)
        if diff >= ERA_THRESHOLD:
            if home_era < away_era:
                txt = (
                    f"✦ {_name(home_sp)} (home SP) has a {diff:.2f} ERA edge "
                    f"({home_era:.2f} TT-ERA vs {away_era:.2f}) — significant pitching advantage"
                )
            else:
                txt = (
                    f"✦ {_name(away_sp)} (away SP) has a {diff:.2f} ERA edge "
                    f"({away_era:.2f} TT-ERA vs {home_era:.2f}) — pitching favors away team"
                )
            factors.append((diff * 10, txt))  # Scale for ranking

    # ── 2. SP K-rate differential (SwStr% + CSW%) ─────────────────────────────
    if home_sp and away_sp:
        home_swstr = getattr(home_sp, "swstr_pct", 0.115)
        away_swstr = getattr(away_sp, "swstr_pct", 0.115)
        home_csw   = getattr(home_sp, "csw_pct",   0.280)
        away_csw   = getattr(away_sp, "csw_pct",   0.280)
        swstr_diff = abs(home_swstr - away_swstr)
        csw_diff   = abs(home_csw   - away_csw)

        if swstr_diff >= 0.015 or csw_diff >= 0.020:
            if home_swstr > away_swstr:
                txt = (
                    f"✦ {_name(home_sp)} has better swing-and-miss (SwStr% {home_swstr*100:.1f}% "
                    f"vs {away_swstr*100:.1f}%) — strikeout edge for home SP"
                )
            else:
                txt = (
                    f"✦ {_name(away_sp)} has better swing-and-miss (SwStr% {away_swstr*100:.1f}% "
                    f"vs {home_swstr*100:.1f}%) — strikeout edge for away SP"
                )
            factors.append((swstr_diff * 60, txt))

    # ── 3. Lineup xwOBA differential ──────────────────────────────────────────
    home_xwoba = _avg_lineup_xwoba(game_context.home_lineup)
    away_xwoba = _avg_lineup_xwoba(game_context.away_lineup)
    xwoba_diff = abs(home_xwoba - away_xwoba)

    if xwoba_diff >= XWOBA_THRESHOLD:
        if home_xwoba > away_xwoba:
            txt = (
                f"✦ Home lineup avg xwOBA {home_xwoba:.3f} vs away {away_xwoba:.3f} "
                f"(+{xwoba_diff:.3f}) — stronger home offense"
            )
        else:
            txt = (
                f"✦ Away lineup avg xwOBA {away_xwoba:.3f} vs home {home_xwoba:.3f} "
                f"(+{xwoba_diff:.3f}) — stronger away offense"
            )
        factors.append((xwoba_diff * 100, txt))

    # ── 4. Bullpen xFIP differential ──────────────────────────────────────────
    if home_bp and away_bp:
        home_xfip = getattr(home_bp, "xfip", 4.20)
        away_xfip = getattr(away_bp, "xfip", 4.20)
        bp_diff   = abs(home_xfip - away_xfip)

        if bp_diff >= BULLPEN_THRESHOLD:
            if home_xfip < away_xfip:
                txt = (
                    f"✦ Home bullpen xFIP {home_xfip:.2f} vs away {away_xfip:.2f} "
                    f"— late-inning advantage for home team"
                )
            else:
                txt = (
                    f"✦ Away bullpen xFIP {away_xfip:.2f} vs home {home_xfip:.2f} "
                    f"— late-inning advantage for away team"
                )
            factors.append((bp_diff * 5, txt))

    # ── 5. Bullpen fatigue flags ───────────────────────────────────────────────
    home_fatigue = getattr(home_bp, "fatigue_flag", False) if home_bp else False
    away_fatigue = getattr(away_bp, "fatigue_flag", False) if away_bp else False

    if away_fatigue and not home_fatigue:
        txt = "✦ Away bullpen fatigue flag — pitched extensively prior night (extra innings or heavy usage)"
        factors.append((4.0, txt))
    elif home_fatigue and not away_fatigue:
        txt = "✦ Home bullpen fatigue flag — pitched extensively prior night (extra innings or heavy usage)"
        factors.append((4.0, txt))
    elif home_fatigue and away_fatigue:
        txt = "✦ Both bullpens flagged for fatigue — expect volatile late-game relief situations"
        factors.append((3.0, txt))

    # ── 6. Recent form (decay win%) ────────────────────────────────────────────
    home_wp = getattr(game_context, "home_decay_win_pct", None)
    away_wp = getattr(game_context, "away_decay_win_pct", None)

    if home_wp is not None and away_wp is not None:
        wp_diff = abs(home_wp - away_wp)
        if wp_diff >= WIN_PCT_THRESHOLD:
            if home_wp > away_wp:
                txt = (
                    f"✦ Home team recent form: {home_wp*100:.1f}% vs away {away_wp*100:.1f}% "
                    f"win% (last 15 games, exp-decay weighted)"
                )
            else:
                txt = (
                    f"✦ Away team recent form: {away_wp*100:.1f}% vs home {home_wp*100:.1f}% "
                    f"win% (last 15 games, exp-decay weighted)"
                )
            factors.append((wp_diff * 50, txt))

    # ── 7. Weather impact ─────────────────────────────────────────────────────
    if park:
        weather_adj = getattr(park, "net_run_adj",
                             getattr(park, "weather_run_adj", 0.0))
        dome        = getattr(park, "is_dome", False)

        if not dome and abs(weather_adj) >= WEATHER_THRESHOLD:
            direction = "hitter-friendly" if weather_adj > 0 else "pitcher-friendly"
            sign      = "+" if weather_adj > 0 else ""
            txt = (
                f"✦ Weather impact: {sign}{weather_adj:.1f} runs/game net adjustment "
                f"({direction} conditions today)"
            )
            factors.append((abs(weather_adj) * 3, txt))

    # ── 8. Park factor ────────────────────────────────────────────────────────
    if park:
        run_factor = getattr(park, "run_factor", 1.00)
        hr_factor  = getattr(park, "hr_factor",  1.00)
        run_dev    = abs(run_factor - 1.00)

        if run_dev >= PARK_THRESHOLD:
            park_name = getattr(park, "venue_name",
                            getattr(park, "stadium_name", "this park"))
            if run_factor > 1.0:
                txt = (
                    f"✦ Park factor {run_factor:.3f} runs / {hr_factor:.3f} HR "
                    f"({park_name} plays as a hitter's park)"
                )
            else:
                txt = (
                    f"✦ Park factor {run_factor:.3f} runs / {hr_factor:.3f} HR "
                    f"({park_name} suppresses run scoring)"
                )
            factors.append((run_dev * 20, txt))

    # ── 9. Elo differential ───────────────────────────────────────────────────
    home_elo = getattr(game_context, "home_elo", None)
    away_elo = getattr(game_context, "away_elo", None)

    if home_elo and away_elo:
        elo_diff = abs(home_elo - away_elo)
        if elo_diff >= ELO_THRESHOLD:
            if home_elo > away_elo:
                txt = (
                    f"✦ Home team Elo rating {home_elo:.0f} vs away {away_elo:.0f} "
                    f"(+{elo_diff:.0f} pts — stronger overall team strength rating)"
                )
            else:
                txt = (
                    f"✦ Away team Elo rating {away_elo:.0f} vs home {home_elo:.0f} "
                    f"(+{elo_diff:.0f} pts — stronger overall team strength rating)"
                )
            factors.append((elo_diff * 0.5, txt))

    # ── 10. Home field advantage note (always include) ────────────────────────
    factors.append((1.0, "✦ Home field advantage: +3.5% baseline (encoded in Elo system + simulation)"))

    # ── Sort by magnitude, return top N ──────────────────────────────────────
    factors.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in factors[:max_factors]]


def weather_summary(park) -> dict:
    """
    Build a weather display dict for the report's weather section.

    Parameters
    ----------
    park : ParkContext object with weather fields.

    Returns
    -------
    dict with keys:
      temp_str     : str — e.g., "72°F → 0.0 runs/game"
      wind_str     : str — e.g., "14mph OUT to CF → +0.4 runs/game"
      dome_flag    : bool
      net_adj_str  : str — e.g., "+0.4 runs/game (slight hitter-friendly)"
      conditions   : str — "DOME", "HITTER-FRIENDLY", "PITCHER-FRIENDLY", "NEUTRAL"
    """
    if park is None:
        return {
            "temp_str":   "N/A",
            "wind_str":   "N/A",
            "dome_flag":  False,
            "net_adj_str": "N/A",
            "conditions": "UNKNOWN",
        }

    dome        = getattr(park, "is_dome", False)
    retractable = getattr(park, "is_retractable", False)
    temp_f      = getattr(park, "temp_f", getattr(park, "temperature_f", None))
    wind_mph    = getattr(park, "wind_speed_mph", 0.0)
    # wind_classification: "out", "in", "cross", "calm"
    wind_dir    = getattr(park, "wind_classification",
                          getattr(park, "wind_direction_label", "calm"))
    temp_adj    = getattr(park, "temp_run_adj", 0.0)
    wind_adj    = getattr(park, "wind_run_adj", 0.0)
    # net_run_adj is the canonical field name in ParkContext
    net_adj     = getattr(park, "net_run_adj",
                          getattr(park, "weather_run_adj", 0.0))

    if dome or retractable:
        return {
            "temp_str":   "N/A (dome/retractable)",
            "wind_str":   "N/A (dome/retractable)",
            "dome_flag":  True,
            "net_adj_str": "0.0 runs (dome — weather neutral)",
            "conditions": "DOME",
        }

    # Temperature string
    if temp_f is not None:
        sign = "+" if temp_adj > 0 else ""
        temp_str = f"{temp_f:.0f}°F → {sign}{temp_adj:.1f} runs/game"
    else:
        temp_str = "Temperature unavailable"

    # Wind string
    if wind_mph < 1:
        wind_str = "Calm (<1 mph) → 0.0 runs/game"
    else:
        sign = "+" if wind_adj > 0 else ""
        wind_str = f"{wind_mph:.0f}mph {wind_dir.upper()} → {sign}{wind_adj:.1f} runs/game"

    # Net adjustment string
    if abs(net_adj) < 0.15:
        conditions = "NEUTRAL"
        condition_label = "neutral conditions"
    elif net_adj > 0:
        conditions = "HITTER-FRIENDLY"
        condition_label = "hitter-friendly conditions"
    else:
        conditions = "PITCHER-FRIENDLY"
        condition_label = "pitcher-friendly conditions"

    sign = "+" if net_adj >= 0 else ""
    net_adj_str = f"{sign}{net_adj:.1f} runs/game ({condition_label})"

    return {
        "temp_str":    temp_str,
        "wind_str":    wind_str,
        "dome_flag":   False,
        "net_adj_str": net_adj_str,
        "conditions":  conditions,
    }


def lineup_confidence_summary(lineup: list) -> str:
    """
    Return "CONFIRMED", "PROJECTED", or "MIXED" based on the is_projected flags
    on each BatterProfile in the lineup.

    Parameters
    ----------
    lineup : list[BatterProfile]

    Returns
    -------
    str — "CONFIRMED", "PROJECTED", or "MIXED"
    """
    if not lineup:
        return "UNKNOWN"

    projected_count = sum(1 for b in lineup if getattr(b, "is_projected", True))
    confirmed_count = len(lineup) - projected_count

    if confirmed_count == len(lineup):
        return "CONFIRMED"
    elif projected_count == len(lineup):
        return "PROJECTED"
    else:
        return "MIXED"


def pitcher_confidence_label(pitcher) -> str:
    """
    Return a display label for a pitcher's data confidence level.
    HIGH / MEDIUM / LOW based on BF count and is_on_il flag.
    """
    if pitcher is None:
        return "TBD"
    confidence = getattr(pitcher, "confidence", "LOW")
    on_il      = getattr(pitcher, "is_on_il", False)
    if on_il:
        return "⚠ IL RETURN"
    return confidence


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal (0–1) as percentage string: 0.228 → '22.8%'"""
    return f"{value * 100:.{decimals}f}%"


def format_era(value: Optional[float]) -> str:
    """Format an ERA value: 4.21 → '4.21' | None → 'N/A'"""
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def format_xwoba(value: Optional[float]) -> str:
    """Format xwOBA: 0.321 → '.321' | None → 'N/A'"""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def era_delta_class(home_era: float, away_era: float) -> str:
    """Return CSS class based on ERA comparison (for HTML color coding)."""
    diff = home_era - away_era
    if diff < -0.5:
        return "advantage-home"
    elif diff > 0.5:
        return "advantage-away"
    return "neutral"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _avg_lineup_xwoba(lineup: list) -> float:
    """Average xwOBA across a lineup."""
    if not lineup:
        return 0.320
    vals = [getattr(b, "xwoba", 0.320) for b in lineup]
    return sum(vals) / len(vals) if vals else 0.320


def _name(pitcher) -> str:
    """Return pitcher name or 'SP' fallback."""
    return getattr(pitcher, "player_name", "SP") or "SP"
