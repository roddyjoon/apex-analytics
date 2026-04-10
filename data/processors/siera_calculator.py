"""
Apex Analytics — True SIERA Calculator
Computes SIERA (Skill-Interactive ERA) from raw Statcast inputs.
Formula: Zimmermann (2010) — the standard in baseball analytics.

SIERA is the best forward-looking ERA estimator because:
  - Accounts for K%, BB%, and batted ball types simultaneously
  - Captures non-linear interactions (K rate × GB rate)
  - More predictive than FIP, xFIP, or ERA for future performance
"""

import logging
from typing import Optional

from config import (
    SIERA_CONST,
    SIERA_K_COEF,
    SIERA_BB_COEF,
    SIERA_INTERACTION_COEF,
    SIERA_GB_FB_SQ_COEF,
    SIERA_LD_SQ_COEF,
    SIERA_GB_FB_K_COEF,
    SIERA_GB_FB_COEF,
    LEAGUE_AVG_ERA,
)

logger = logging.getLogger(__name__)

# Reasonable bounds for SIERA output
SIERA_MIN = 1.50
SIERA_MAX = 9.00


def compute_siera(
    k_pct: float,
    bb_pct: float,
    gb_pct: float,
    fb_pct: float,
    ld_pct: float,
) -> float:
    """
    Compute true SIERA from raw batted-ball and plate-discipline inputs.

    All inputs as decimals (0.0–1.0), not percentages:
      k_pct  : Strikeout rate (K / PA)
      bb_pct : Walk rate (BB / PA)
      gb_pct : Ground ball rate (GB / BIP)
      fb_pct : Fly ball rate (FB / BIP)
      ld_pct : Line drive rate (LD / BIP)

    Returns SIERA as a float (ERA scale, typically 2.5–6.5).
    """
    # Validate inputs
    k_pct, bb_pct, gb_pct, fb_pct, ld_pct = _sanitize_inputs(k_pct, bb_pct, gb_pct, fb_pct, ld_pct)

    # GB−FB differential (key non-linear term in formula)
    gb_fb_diff = gb_pct - fb_pct

    # Zimmermann SIERA formula (2010)
    siera = (
        SIERA_CONST
        + SIERA_K_COEF          * k_pct
        + SIERA_BB_COEF         * bb_pct
        + SIERA_INTERACTION_COEF * (gb_fb_diff * k_pct)
        + SIERA_GB_FB_SQ_COEF   * (gb_fb_diff ** 2)
        + SIERA_LD_SQ_COEF      * (ld_pct ** 2)
        + SIERA_GB_FB_K_COEF    * (gb_fb_diff * (1 - k_pct))
        + SIERA_GB_FB_COEF      * gb_fb_diff
    )

    siera = float(max(SIERA_MIN, min(SIERA_MAX, siera)))
    logger.debug(
        "SIERA: %.3f  [K=%.3f, BB=%.3f, GB=%.3f, FB=%.3f, LD=%.3f]",
        siera, k_pct, bb_pct, gb_pct, fb_pct, ld_pct
    )
    return siera


def compute_fip(
    k_pct: float,
    bb_pct: float,
    hr_rate: float,          # HR per batters faced (not per 9)
    fip_constant: float,     # Season FIP constant (from config: 3.10 for 2024)
) -> float:
    """
    Compute FIP (Fielding Independent Pitching).

    FIP = ((13×HR) + (3×(BB+HBP)) − (2×K)) / IP + FIP_constant
    Converted here from rate stats for easier integration.

    Formula in rate form:
      FIP_rate = 13×hr_rate + 3×bb_pct − 2×k_pct
      FIP ≈ FIP_rate × scaling + constant

    Using the standard PA-based approximation:
      FIP = (13*HR_per_BF*9/1 + 3*bb_pct*9 - 2*k_pct*9) + C
    This gives ERA-scale output.
    """
    k_pct   = max(0.0, min(0.60, k_pct))
    bb_pct  = max(0.0, min(0.30, bb_pct))
    hr_rate = max(0.0, min(0.15, hr_rate))

    fip = (13 * hr_rate * 9) + (3 * bb_pct * 9) - (2 * k_pct * 9) + fip_constant
    return float(max(SIERA_MIN, min(SIERA_MAX, fip)))


def blend_true_talent_era(
    siera: float,
    xera: float,
    fip: float,
    siera_weight: float = 0.45,
    xera_weight:  float = 0.35,
    fip_weight:   float = 0.20,
) -> float:
    """
    Blend SIERA, xERA, and FIP into a single true-talent ERA estimate.

    Default weights (from config):
      SIERA × 0.45 — best forward-looking metric; highest weight
      xERA  × 0.35 — Statcast contact quality; second highest
      FIP   × 0.20 — stable floor; lowest weight

    All three inputs should be on ERA scale.
    """
    total_w = siera_weight + xera_weight + fip_weight
    if abs(total_w - 1.0) > 0.01:
        # Normalize weights if they don't sum to 1.0
        siera_weight /= total_w
        xera_weight  /= total_w
        fip_weight   /= total_w

    blended = (siera * siera_weight) + (xera * xera_weight) + (fip * fip_weight)
    return float(max(SIERA_MIN, min(SIERA_MAX, blended)))


def compute_stuff_plus_proxy(swstr_pct: float, league_avg_swstr: float = 0.115) -> float:
    """
    Simple Stuff+ proxy based on SwStr% relative to league average.
    100 = league average. >110 = above average stuff.

    Used in starter removal probability model (aces stay longer).
    """
    if league_avg_swstr <= 0:
        return 100.0
    return round((swstr_pct / league_avg_swstr) * 100, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_inputs(
    k_pct: float, bb_pct: float,
    gb_pct: float, fb_pct: float, ld_pct: float
) -> tuple[float, float, float, float, float]:
    """Clamp all inputs to valid ranges and handle zero/missing values."""

    k_pct  = _clamp(k_pct,  0.05, 0.55, default=0.228)
    bb_pct = _clamp(bb_pct, 0.02, 0.25, default=0.083)

    # Batted ball rates should sum close to 1.0
    gb_pct = _clamp(gb_pct, 0.10, 0.80, default=0.430)
    fb_pct = _clamp(fb_pct, 0.10, 0.70, default=0.350)
    ld_pct = _clamp(ld_pct, 0.10, 0.40, default=0.220)

    # Normalize batted ball rates to sum to 1.0 (they come from BIP)
    bip_total = gb_pct + fb_pct + ld_pct
    if bip_total > 0 and abs(bip_total - 1.0) > 0.05:
        gb_pct /= bip_total
        fb_pct /= bip_total
        ld_pct /= bip_total

    return k_pct, bb_pct, gb_pct, fb_pct, ld_pct


def _clamp(val: Optional[float], lo: float, hi: float, default: float) -> float:
    if val is None or not isinstance(val, (int, float)):
        return default
    try:
        v = float(val)
        if v != v:  # NaN check
            return default
        return max(lo, min(hi, v))
    except (ValueError, TypeError):
        return default
