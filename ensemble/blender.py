"""
Apex Analytics — Four-Layer Ensemble Blender
Combines Monte Carlo, Elo, Random Forest, and Logistic Regression outputs
into a single calibrated win probability.

Architecture:
  1. Monte Carlo simulation (physics-based, Statcast-driven, bottom-up)
  2. Elo rating system (dynamic team strength, post-game updated)
  3. Random Forest (non-linear feature interactions)
  4. Logistic Regression (interpretable linear baseline)

Weights are adjusted by season phase:
  - April (Opening Day → Apr 15): MC=30%, Elo=15%, RF=30%, LR=25%
    Rationale: tiny stat samples; RF/LR base rates + prior-season signal dominate
  - Late April – May (Apr 16 → May 31): MC=45%, Elo=12%, RF=25%, LR=18%
    Rationale: growing sample; MC gaining accuracy; RF still useful
  - June – July: MC=60%, Elo=10%, RF=20%, LR=10%
    Rationale: full season stats; MC is primary; RF catches non-linear patterns
  - Aug – Sept: MC=65%, Elo=8%, RF=18%, LR=9%
    Rationale: deep samples; MC peak reliability; Elo near-irrelevant vs. Statcast

Output: calibrated win probability (0.0–1.0) for home team + full breakdown dict.
"""

import logging
from datetime import date, datetime
from typing import Optional

import numpy as np

# Season-phase weights are defined in _SEASON_PHASE_WEIGHTS table below

logger = logging.getLogger(__name__)

# Season-phase weight tables [mc, elo, rf, lr]
# Indexed by (month_start, month_end_inclusive) → weights
_SEASON_PHASE_WEIGHTS: list[dict] = [
    # Phase 1: Opening Day – April 15
    {
        "label":      "Early April (small sample)",
        "date_start": (3, 28),   # Mar 28 (earliest Opening Day)
        "date_end":   (4, 15),
        "mc":  0.30,
        "elo": 0.15,
        "rf":  0.30,
        "lr":  0.25,
    },
    # Phase 2: Late April – May
    {
        "label":      "Late April – May",
        "date_start": (4, 16),
        "date_end":   (5, 31),
        "mc":  0.45,
        "elo": 0.12,
        "rf":  0.25,
        "lr":  0.18,
    },
    # Phase 3: June – July
    {
        "label":      "June – July",
        "date_start": (6, 1),
        "date_end":   (7, 31),
        "mc":  0.60,
        "elo": 0.10,
        "rf":  0.20,
        "lr":  0.10,
    },
    # Phase 4: August – September (default for anything else too)
    {
        "label":      "August – September",
        "date_start": (8, 1),
        "date_end":   (10, 5),
        "mc":  0.65,
        "elo": 0.08,
        "rf":  0.18,
        "lr":  0.09,
    },
]

# Fallback weights (used when date is ambiguous or off-season)
_DEFAULT_WEIGHTS = {"mc": 0.60, "elo": 0.10, "rf": 0.20, "lr": 0.10}


def blend_predictions(
    mc_prob:    float,
    elo_prob:   float,
    rf_prob:    float,
    lr_prob:    float,
    game_date:  Optional[date] = None,
    apply_calibration: bool = True,
) -> dict:
    """
    Combine four model outputs into a single calibrated win probability.

    Parameters
    ----------
    mc_prob     : Monte Carlo home win probability (0.0–1.0).
    elo_prob    : Elo home win probability (0.0–1.0).
    rf_prob     : Random Forest home win probability (0.0–1.0).
    lr_prob     : Logistic Regression home win probability (0.0–1.0).
    game_date   : Date of the game (used for season-phase weight selection).
                  Defaults to today if None.
    apply_calibration : Whether to apply the fitted calibration layer.

    Returns
    -------
    dict with keys:
      mc_prob          : float — Monte Carlo input
      elo_prob         : float — Elo input
      rf_prob          : float — RF input
      lr_prob          : float — LR input
      weights          : dict — {mc, elo, rf, lr} weights used
      phase_label      : str — season phase description
      raw_ensemble     : float — weighted average before calibration
      calibrated_prob  : float — final calibrated home win probability
      away_prob        : float — 1 - calibrated_prob
      confidence_band  : tuple — (lower, upper) ±1 std-dev range
    """
    if game_date is None:
        game_date = date.today()

    # Clamp all inputs to valid probability range
    mc_prob  = float(np.clip(mc_prob,  0.0, 1.0))
    elo_prob = float(np.clip(elo_prob, 0.0, 1.0))
    rf_prob  = float(np.clip(rf_prob,  0.0, 1.0))
    lr_prob  = float(np.clip(lr_prob,  0.0, 1.0))

    # Get season-phase weights
    weights, phase_label = _get_phase_weights(game_date)

    # Weighted ensemble
    raw_ensemble = (
        mc_prob  * weights["mc"]  +
        elo_prob * weights["elo"] +
        rf_prob  * weights["rf"]  +
        lr_prob  * weights["lr"]
    )
    raw_ensemble = float(np.clip(raw_ensemble, 0.05, 0.95))

    # Calibration
    if apply_calibration:
        from ensemble.calibrator import get_calibrator
        calibrator = get_calibrator()
        if calibrator.is_fitted:
            calibrated_prob = calibrator.calibrate_single(raw_ensemble)
        else:
            calibrated_prob = raw_ensemble
    else:
        calibrated_prob = raw_ensemble

    calibrated_prob = float(np.clip(calibrated_prob, 0.05, 0.95))

    # Confidence band: ±1 weighted std-dev of the four models
    probs  = np.array([mc_prob, elo_prob, rf_prob, lr_prob])
    ws     = np.array([weights["mc"], weights["elo"], weights["rf"], weights["lr"]])
    std_dev = float(np.sqrt(np.average((probs - raw_ensemble) ** 2, weights=ws)))
    band_lower = float(np.clip(calibrated_prob - std_dev, 0.01, 0.99))
    band_upper = float(np.clip(calibrated_prob + std_dev, 0.01, 0.99))

    logger.debug(
        "Blend: MC=%.3f(%.0f%%) Elo=%.3f(%.0f%%) RF=%.3f(%.0f%%) "
        "LR=%.3f(%.0f%%) → raw=%.3f → cal=%.3f [phase: %s]",
        mc_prob,  weights["mc"]  * 100,
        elo_prob, weights["elo"] * 100,
        rf_prob,  weights["rf"]  * 100,
        lr_prob,  weights["lr"]  * 100,
        raw_ensemble, calibrated_prob, phase_label,
    )

    return {
        "mc_prob":         round(mc_prob,         4),
        "elo_prob":        round(elo_prob,         4),
        "rf_prob":         round(rf_prob,          4),
        "lr_prob":         round(lr_prob,          4),
        "weights":         {k: round(v, 4) for k, v in weights.items()},
        "phase_label":     phase_label,
        "raw_ensemble":    round(raw_ensemble,     4),
        "calibrated_prob": round(calibrated_prob,  4),
        "away_prob":       round(1.0 - calibrated_prob, 4),
        "confidence_band": (round(band_lower, 4), round(band_upper, 4)),
    }


def blend_from_context(
    mc_result:    dict,
    game_context,
    game_date:    Optional[date] = None,
    apply_calibration: bool = True,
) -> dict:
    """
    Higher-level blend function that accepts simulation + context objects
    and builds all four layer predictions internally.

    Parameters
    ----------
    mc_result    : Output dict from monte_carlo.run_monte_carlo().
    game_context : GameContext object with full profiles for both teams.
    game_date    : Date of the game.

    Returns
    -------
    dict — same as blend_predictions() plus additional context fields.
    """
    mc_prob = mc_result.get("home_win_pct", 0.53)

    # Elo prediction
    elo_prob = _get_elo_prob(game_context)

    # Random Forest prediction
    rf_prob = _get_rf_prob(game_context)

    # Logistic Regression prediction
    lr_prob = _get_lr_prob(game_context)

    result = blend_predictions(
        mc_prob=mc_prob,
        elo_prob=elo_prob,
        rf_prob=rf_prob,
        lr_prob=lr_prob,
        game_date=game_date,
        apply_calibration=apply_calibration,
    )

    # Attach simulation metadata
    result["mc_iterations"]  = mc_result.get("n_iterations", 0)
    result["mc_elapsed_sec"] = mc_result.get("elapsed_seconds", 0.0)
    result["mc_ci"]          = mc_result.get("confidence_interval", (0.0, 1.0))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_phase_weights(game_date: date) -> tuple[dict, str]:
    """
    Return the season-phase weight dict and label for a given game date.
    Matches against _SEASON_PHASE_WEIGHTS table; falls back to default.
    """
    month = game_date.month
    day   = game_date.day

    for phase in _SEASON_PHASE_WEIGHTS:
        sm, sd = phase["date_start"]
        em, ed = phase["date_end"]

        # Convert to a single comparable int for range check
        game_md  = month * 100 + day
        start_md = sm    * 100 + sd
        end_md   = em    * 100 + ed

        if start_md <= game_md <= end_md:
            weights = {
                "mc":  phase["mc"],
                "elo": phase["elo"],
                "rf":  phase["rf"],
                "lr":  phase["lr"],
            }
            return weights, phase["label"]

    logger.debug(
        "Game date %s not in any season phase — using default weights.", game_date
    )
    return _DEFAULT_WEIGHTS.copy(), "Default (mid-season)"


def _get_elo_prob(context) -> float:
    """Extract Elo win probability from GameContext."""
    try:
        from ensemble.elo_system import win_probability, get_team_elo

        home_elo = getattr(context, "home_elo", None)
        away_elo = getattr(context, "away_elo", None)

        if home_elo is None or away_elo is None:
            # Try to load from DB
            season = int(context.game_date[:4]) if context.game_date else date.today().year
            home_elo = get_team_elo(context.home_team_id, season)
            away_elo = get_team_elo(context.away_team_id, season)

        return win_probability(home_elo, away_elo)

    except Exception as exc:
        logger.warning("Elo probability failed: %s — using 0.53 default.", exc)
        return 0.53


def _get_rf_prob(context) -> float:
    """Build RF feature dict from GameContext and predict."""
    try:
        from ensemble.random_forest_model import get_rf_model

        features = _extract_rf_features(context)
        return get_rf_model().predict_win_probability(features)

    except Exception as exc:
        logger.warning("RF probability failed: %s — using 0.53 default.", exc)
        return 0.53


def _get_lr_prob(context) -> float:
    """Build LR feature dict from GameContext and predict."""
    try:
        from ensemble.logistic_model import get_lr_model

        features = _extract_lr_features(context)
        return get_lr_model().predict_win_probability(features)

    except Exception as exc:
        logger.warning("LR probability failed: %s — using 0.53 default.", exc)
        return 0.53


def _extract_rf_features(context) -> dict:
    """
    Extract the 15 RF features from a GameContext.
    Falls back to defaults for any missing field.
    """
    home_sp  = context.home_starter
    away_sp  = context.away_starter
    home_bp  = context.home_bullpen
    away_bp  = context.away_bullpen
    park     = context.park

    # Team-level lineup xwOBA (average across the lineup)
    home_xwoba = _lineup_avg_xwoba(context.home_lineup)
    away_xwoba = _lineup_avg_xwoba(context.away_lineup)

    # Team decay win%
    home_decay_wp = getattr(context, "home_decay_win_pct", 0.500)
    away_decay_wp = getattr(context, "away_decay_win_pct", 0.500)

    # Elo
    home_elo = getattr(context, "home_elo", 1500.0)
    away_elo = getattr(context, "away_elo", 1500.0)

    return {
        "home_starter_siera":       getattr(home_sp, "siera", 4.20),
        "away_starter_siera":       getattr(away_sp, "siera", 4.20),
        "home_starter_decay_xera":  getattr(home_sp, "exp_decay_xera", 4.20) or 4.20,
        "away_starter_decay_xera":  getattr(away_sp, "exp_decay_xera", 4.20) or 4.20,
        "home_team_xwoba":          home_xwoba,
        "away_team_xwoba":          away_xwoba,
        "home_bullpen_xfip":        getattr(home_bp, "xfip", 4.20),
        "away_bullpen_xfip":        getattr(away_bp, "xfip", 4.20),
        "park_factor_runs":         getattr(park, "run_factor", 1.00),
        "park_factor_hr":           getattr(park, "hr_factor",  1.00),
        "home_decay_win_pct":       home_decay_wp,
        "away_decay_win_pct":       away_decay_wp,
        "weather_run_adj":          getattr(park, "weather_run_adj", 0.0),
        "home_elo":                 home_elo,
        "away_elo":                 away_elo,
    }


def _extract_lr_features(context) -> dict:
    """
    Extract the 12 LR features from a GameContext.
    Falls back to defaults for any missing field.
    """
    home_sp  = context.home_starter
    away_sp  = context.away_starter
    home_bp  = context.home_bullpen
    away_bp  = context.away_bullpen
    park     = context.park

    home_xwoba = _lineup_avg_xwoba(context.home_lineup)
    away_xwoba = _lineup_avg_xwoba(context.away_lineup)

    home_decay_wp = getattr(context, "home_decay_win_pct", 0.500)
    away_decay_wp = getattr(context, "away_decay_win_pct", 0.500)

    return {
        "home_starter_era":        getattr(home_sp, "true_talent_era", 4.20),
        "away_starter_era":        getattr(away_sp, "true_talent_era", 4.20),
        "home_team_xwoba":         home_xwoba,
        "away_team_xwoba":         away_xwoba,
        "home_bullpen_xfip":       getattr(home_bp, "xfip", 4.20),
        "away_bullpen_xfip":       getattr(away_bp, "xfip", 4.20),
        "park_factor_runs":        getattr(park, "run_factor", 1.00),
        "home_decay_win_pct":      home_decay_wp,
        "away_decay_win_pct":      away_decay_wp,
        "weather_run_adj":         getattr(park, "weather_run_adj", 0.0),
        "home_starter_decay_xera": getattr(home_sp, "exp_decay_xera", 4.20) or 4.20,
        "away_starter_decay_xera": getattr(away_sp, "exp_decay_xera", 4.20) or 4.20,
    }


def _lineup_avg_xwoba(lineup: list) -> float:
    """Compute average xwOBA across a lineup. Defaults to 0.320 if empty."""
    if not lineup:
        return 0.320
    values = [getattr(b, "xwoba", 0.320) for b in lineup]
    return float(np.mean(values)) if values else 0.320


def get_phase_weights_for_date(game_date: Optional[date] = None) -> dict:
    """
    Public helper: return the ensemble weights active on a given date.
    Useful for reporting and transparency layer.
    """
    if game_date is None:
        game_date = date.today()
    weights, label = _get_phase_weights(game_date)
    weights["phase_label"] = label
    return weights
