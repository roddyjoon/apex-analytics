"""
Apex Analytics — Plate Appearance Outcome Calculator
9-outcome probability engine for the Monte Carlo simulation.

Outcomes: hr, triple, double, single, walk, hbp, strikeout, groundout, flyout

Adjustment chain (applied in order, normalized at end):
  a. League baseline (2024 MLB averages)
  b. Batter multiplier — Bayesian-blended xwOBA (exp_decay if available)
  c. Pitcher suppression — true_talent_era → ERA-scale suppression factor
  d. SIERA K-rate modifier — CSW% + SwStr% vs league avg
  e. Platoon adjustment — batter xwOBA vs pitcher handedness
  f. Pitch-type vulnerability — arsenal-weighted xwOBA multiplier
  g. Batter-vs-pitcher matchup override — head-to-head PA history
  h. Park factor — HR factor for fly balls, run factor for ground balls
  i. Weather adjustment — net_run_adj from ParkContext
  j. Normalize to sum exactly 1.0
"""

import logging
from typing import Optional

import numpy as np

from config import (
    LEAGUE_AVG_XWOBA,
    LEAGUE_AVG_K_RATE,
    LEAGUE_AVG_BB_RATE,
    LEAGUE_AVG_HR_RATE,
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_SWSTR_PCT,
    LEAGUE_AVG_CSW_PCT,
    LEAGUE_AVG_BARREL_PCT,
)
from data.processors.pitch_vulnerability import (
    compute_pitch_vulnerability_adj,
    compute_k_rate_vulnerability_adj,
)
from data.processors.matchup_adjuster import get_matchup_xwoba_multiplier
from simulation.profiles import BatterProfile, PitcherProfile, ParkContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2024 MLB League Baseline Outcome Rates (per PA)
# Source: Baseball Reference 2024 league totals
# ---------------------------------------------------------------------------
LEAGUE_BASELINE = {
    "hr":        0.037,   # 3.7% — HR per PA
    "triple":    0.005,   # 0.5%
    "double":    0.047,   # 4.7%
    "single":    0.149,   # 14.9%
    "walk":      0.083,   # 8.3% (BB + IBB)
    "hbp":       0.011,   # 1.1%
    "strikeout": 0.228,   # 22.8%
    "groundout": 0.240,   # 24.0% (includes GIDP probability)
    "flyout":    0.200,   # 20.0% (includes sac fly)
}
# Sum ≈ 1.000

# Minimum/maximum clamps for probabilities
_MIN_PROB = 0.001
_MAX_HR   = 0.120
_MAX_K    = 0.500
_MAX_BB   = 0.250

# ERA-to-suppression scaling: how much does ERA deviate from league avg affect outcomes
_ERA_SCALE = 0.08   # Per 1.0 ERA unit above/below league avg

# xwOBA-to-multiplier scale: how much does batter xwOBA drive outcome scaling
_XWOBA_SCALE = 1.0  # Direct ratio: batter_xwoba / league_avg_xwoba


def compute_pa_outcomes(
    batter:  BatterProfile,
    pitcher: PitcherProfile,
    park:    Optional[ParkContext] = None,
    outs:    int = 0,
) -> dict[str, float]:
    """
    Compute plate appearance outcome probabilities for one batter vs. one pitcher.

    Parameters
    ----------
    batter  : BatterProfile — fully enriched batter object.
    pitcher : PitcherProfile — fully enriched pitcher object (with fatigue applied).
    park    : ParkContext — park factors + weather. None = neutral park, no weather.
    outs    : Current outs in the inning (used for groundout/DP adjustment).

    Returns
    -------
    dict[str, float] : Outcome probabilities summing to exactly 1.0.
    Keys: "hr", "triple", "double", "single", "walk", "hbp",
          "strikeout", "groundout", "flyout"
    """
    probs = dict(LEAGUE_BASELINE)  # Start with league baseline

    # ── a. Batter multiplier ─────────────────────────────────────────────────
    # Use exp_decay_xwoba if available (recent form), else season xwOBA
    effective_xwoba = batter.exp_decay_xwoba if batter.exp_decay_xwoba is not None else batter.xwoba
    effective_xwoba = max(0.150, min(0.550, effective_xwoba))

    batter_mult = effective_xwoba / LEAGUE_AVG_XWOBA
    batter_mult = max(0.50, min(2.00, batter_mult))

    # Apply batter multiplier to offensive outcomes
    for outcome in ("hr", "triple", "double", "single"):
        probs[outcome] *= batter_mult
    # Batter BB/HBP scales with walk rate
    bb_mult = batter.bb_pct / LEAGUE_AVG_BB_RATE if LEAGUE_AVG_BB_RATE > 0 else 1.0
    bb_mult = max(0.40, min(2.50, bb_mult))
    probs["walk"] *= bb_mult
    probs["hbp"]  *= bb_mult * 0.5  # HBP less correlated to walk rate

    # K rate adjustment from batter side
    k_mult_batter = batter.k_pct / LEAGUE_AVG_K_RATE if LEAGUE_AVG_K_RATE > 0 else 1.0
    k_mult_batter = max(0.40, min(2.50, k_mult_batter))
    probs["strikeout"] *= k_mult_batter

    # HR specifically: barrel% drives HR above baseline
    barrel_mult = batter.barrel_pct / LEAGUE_AVG_BARREL_PCT if LEAGUE_AVG_BARREL_PCT > 0 else 1.0
    barrel_mult = max(0.20, min(3.00, barrel_mult))
    probs["hr"] *= barrel_mult

    # ── b. Pitcher suppression ────────────────────────────────────────────────
    # true_talent_era > league_avg → pitcher is worse → more offense
    # true_talent_era < league_avg → pitcher is better → less offense
    era_delta  = pitcher.true_talent_era - LEAGUE_AVG_ERA
    era_factor = 1.0 + (era_delta * _ERA_SCALE)
    era_factor = max(0.60, min(1.60, era_factor))

    # Suppression pushes offense up/down
    for outcome in ("hr", "triple", "double", "single", "walk"):
        probs[outcome] *= era_factor
    # K rate moves opposite: better pitcher → more Ks
    probs["strikeout"] /= max(0.50, era_factor)

    # ── c. SIERA K-rate modifier ──────────────────────────────────────────────
    # CSW% (called strike + whiff) is the best K-rate predictor
    csw_ratio  = pitcher.csw_pct / LEAGUE_AVG_CSW_PCT if LEAGUE_AVG_CSW_PCT > 0 else 1.0
    swstr_ratio = pitcher.swstr_pct / LEAGUE_AVG_SWSTR_PCT if LEAGUE_AVG_SWSTR_PCT > 0 else 1.0
    # Blend: 60% CSW, 40% SwStr (CSW is stronger predictor)
    k_rate_mult = (csw_ratio * 0.60) + (swstr_ratio * 0.40)
    k_rate_mult = max(0.50, min(2.00, k_rate_mult))
    probs["strikeout"] *= k_rate_mult

    # ── d. Platoon adjustment ─────────────────────────────────────────────────
    # Use appropriate xwOBA split based on pitcher handedness
    if pitcher.throws == "L":
        platoon_xwoba = batter.xwoba_vs_lhp
    else:
        platoon_xwoba = batter.xwoba_vs_rhp

    if platoon_xwoba > 0 and effective_xwoba > 0:
        # Platoon correction: how much does split xwOBA differ from overall?
        platoon_adj = platoon_xwoba / effective_xwoba
        platoon_adj = max(0.70, min(1.40, platoon_adj))
        for outcome in ("hr", "triple", "double", "single"):
            probs[outcome] *= platoon_adj

    # ── e. Pitch-type vulnerability ───────────────────────────────────────────
    if pitcher.arsenal and batter.pitch_type_xwoba:
        vuln_mult, _ = compute_pitch_vulnerability_adj(
            batter_pitch_xwoba=batter.pitch_type_xwoba,
            pitcher_arsenal=pitcher.arsenal,
        )
        vuln_mult = max(0.60, min(1.60, vuln_mult))
        for outcome in ("hr", "triple", "double", "single"):
            probs[outcome] *= vuln_mult

        # K-rate vulnerability separately
        k_vuln = compute_k_rate_vulnerability_adj(
            batter_pitch_kpct=batter.pitch_type_xwoba,
            pitcher_arsenal=pitcher.arsenal,
        )
        k_vuln = max(0.60, min(1.80, k_vuln))
        probs["strikeout"] *= k_vuln

    # ── f. Batter-vs-pitcher matchup override ────────────────────────────────
    if pitcher.player_id in batter.matchup_history:
        matchup_mult, _ = get_matchup_xwoba_multiplier(
            batter_id=batter.player_id,
            pitcher_id=pitcher.player_id,
            batter_xwoba=effective_xwoba,
        )
        matchup_mult = max(0.60, min(1.60, matchup_mult))
        for outcome in ("hr", "triple", "double", "single"):
            probs[outcome] *= matchup_mult

    # ── g. Park factor ────────────────────────────────────────────────────────
    if park is not None and not park.is_dome:
        # HR and triples: use HR park factor (fly ball carry)
        hr_factor  = max(0.60, min(1.60, park.hr_factor))
        # Singles, doubles: use run park factor (speed/dimensions)
        run_factor = max(0.70, min(1.40, park.run_factor))

        probs["hr"]     *= hr_factor
        probs["triple"] *= hr_factor * 0.80  # Triples partially correlated to HR factor
        probs["double"] *= run_factor
        probs["single"] *= run_factor

        # High park factor → fewer flyouts (more go over fence or off wall)
        probs["flyout"] /= max(0.80, hr_factor)

    # ── h. Weather adjustment ─────────────────────────────────────────────────
    if park is not None and not park.is_dome and abs(park.net_run_adj) > 0.05:
        # Convert run adjustment to probability nudge
        # +1.0 run/game ≈ +0.007 on each hit type (rough approximation)
        weather_hit_adj = 1.0 + (park.net_run_adj * 0.007)
        weather_hit_adj = max(0.85, min(1.15, weather_hit_adj))
        for outcome in ("hr", "double", "single"):
            probs[outcome] *= weather_hit_adj
        # HR specifically benefits more from wind-out
        if park.wind_classification == "out" and park.net_run_adj > 0:
            probs["hr"] *= (1.0 + park.net_run_adj * 0.012)
        elif park.wind_classification == "in" and park.net_run_adj < 0:
            probs["hr"] *= (1.0 + park.net_run_adj * 0.012)

    # ── i. Fatigue adjustment on pitcher ─────────────────────────────────────
    if pitcher.fatigue_index > 0:
        fatigue_hit_adj = 1.0 + (pitcher.fatigue_index * 0.15)
        fatigue_hit_adj = max(1.0, min(1.15, fatigue_hit_adj))
        for outcome in ("hr", "double", "single", "walk"):
            probs[outcome] *= fatigue_hit_adj
        probs["strikeout"] /= fatigue_hit_adj

    # ── j. Clamp individual probabilities ─────────────────────────────────────
    probs["hr"]        = max(_MIN_PROB, min(_MAX_HR,  probs["hr"]))
    probs["strikeout"] = max(_MIN_PROB, min(_MAX_K,   probs["strikeout"]))
    probs["walk"]      = max(_MIN_PROB, min(_MAX_BB,  probs["walk"]))
    for key in probs:
        probs[key] = max(_MIN_PROB, probs[key])

    # ── k. Rebalance outs (groundout/flyout absorb excess probability) ────────
    total_non_out = sum(probs[k] for k in ("hr", "triple", "double", "single", "walk", "hbp"))
    total_outs    = sum(probs[k] for k in ("strikeout", "groundout", "flyout"))
    total         = total_non_out + total_outs

    # Distribute ground/fly out proportionally to reach 1.0
    target_non_out = min(total_non_out / total, 0.65)  # Cap non-out at 65% of PAs
    target_outs    = 1.0 - target_non_out

    if total_non_out > 0:
        non_out_scale = target_non_out / (total_non_out / total)
        for k in ("hr", "triple", "double", "single", "walk", "hbp"):
            probs[k] *= non_out_scale / total

    if total_outs > 0:
        out_scale = target_outs / (total_outs / total)
        for k in ("strikeout", "groundout", "flyout"):
            probs[k] *= out_scale / total

    # ── l. Final normalize to exactly 1.0 ────────────────────────────────────
    probs = _normalize(probs)

    logger.debug(
        "PA outcomes [%s vs %s]: HR=%.3f 1B=%.3f BB=%.3f K=%.3f GO=%.3f FO=%.3f",
        batter.player_name[:15], pitcher.player_name[:15],
        probs["hr"], probs["single"], probs["walk"],
        probs["strikeout"], probs["groundout"], probs["flyout"]
    )
    return probs


def sample_outcome(probs: dict[str, float], rng: np.random.Generator) -> str:
    """
    Sample one PA outcome from the probability distribution.

    Parameters
    ----------
    probs : Output of compute_pa_outcomes().
    rng   : numpy random Generator (passed in from simulator for reproducibility).

    Returns
    -------
    One of: "hr", "triple", "double", "single", "walk", "hbp",
            "strikeout", "groundout", "flyout"
    """
    outcomes = list(probs.keys())
    weights  = [probs[k] for k in outcomes]
    idx = rng.choice(len(outcomes), p=weights)
    return outcomes[idx]


def is_out(outcome: str) -> bool:
    return outcome in ("strikeout", "groundout", "flyout")


def is_hit(outcome: str) -> bool:
    return outcome in ("hr", "triple", "double", "single")


def is_on_base(outcome: str) -> bool:
    return outcome in ("hr", "triple", "double", "single", "walk", "hbp")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities to sum to exactly 1.0."""
    total = sum(probs.values())
    if total <= 0:
        # Fallback to baseline if something went catastrophically wrong
        return dict(LEAGUE_BASELINE)
    return {k: v / total for k, v in probs.items()}
