"""
Apex Analytics — Calibration Layer
Corrects systematic bias in ensemble win probability outputs.

The problem:
  Raw ensemble output is overconfident (like all sports prediction models).
  A model that says 70% might actually mean the team wins only 64%.
  Calibration corrects this:
    Before: Model says 70% → team actually wins 64% (overconfident)
    After:  Model says 70% → team actually wins ~70% (calibrated)

Implementation strategy:
  Phase 1 (< 1,000 games): Platt Scaling
    - Logistic regression fitted on model output vs. actual outcome
    - Works with small samples (50+ games minimum)
    - Handles early-season when we have limited historical data

  Phase 2 (>= 1,000 games): Isotonic Regression
    - Non-parametric, more flexible than Platt
    - Learns the actual calibration curve shape
    - Outperforms Platt when sample is large enough
    - Falls back to Platt if isotonic fails

Calibration monitoring:
  - Brier Score (target < 0.240 on 30-day rolling basis)
  - Calibration curve plotted weekly
  - Alert if 30-day Brier > BRIER_ALERT_THRESHOLD

Model persistence: models/calibrator.pkl
"""

import logging
import pickle
from datetime import date, datetime
from typing import Optional

import numpy as np

from config import MODELS_DIR, ISOTONIC_MIN_GAMES, BRIER_ALERT_THRESHOLD
from data.cache.db import get_session, CalibrationHistory

logger = logging.getLogger(__name__)

CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"


class ProbabilityCalibrator:
    """
    Two-phase probability calibrator.
    Phase 1: Platt Scaling (logistic regression on raw probability).
    Phase 2: Isotonic Regression (after ISOTONIC_MIN_GAMES games).
    """

    def __init__(self):
        self.platt_model   = None
        self.isotonic_model = None
        self._n_games      = 0
        self._method       = "none"       # "none", "platt", "isotonic"
        self._is_fitted    = False
        self._load_if_exists()

    @property
    def method(self) -> str:
        return self._method

    @property
    def n_games(self) -> int:
        return self._n_games

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray) -> dict:
        """
        Fit the calibrator on historical predictions vs. actual outcomes.

        Parameters
        ----------
        raw_probs       : Array of uncalibrated ensemble win probabilities.
        actual_outcomes : Array of 1 (home won) or 0 (home lost).

        Returns
        -------
        dict with fit metrics: method, n_games, brier_score, log_loss.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss

        n = len(raw_probs)
        self._n_games = n

        if n < 30:
            logger.warning("Only %d games for calibration — need at least 30. Skipping.", n)
            return {"method": "none", "n_games": n}

        # Choose method based on sample size
        if n >= ISOTONIC_MIN_GAMES:
            # Phase 2: Isotonic Regression
            try:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(raw_probs, actual_outcomes)
                self.isotonic_model = ir
                self._method        = "isotonic"
                self._is_fitted     = True
                logger.info("Calibrator: isotonic regression fitted on %d games.", n)
            except Exception as exc:
                logger.warning("Isotonic fit failed: %s — falling back to Platt.", exc)
                self._fit_platt(raw_probs, actual_outcomes)
        else:
            # Phase 1: Platt Scaling
            self._fit_platt(raw_probs, actual_outcomes)

        # Compute calibration metrics
        calibrated = self.calibrate(raw_probs)
        brier      = float(brier_score_loss(actual_outcomes, calibrated))
        ll         = float(log_loss(actual_outcomes, calibrated))

        if brier > BRIER_ALERT_THRESHOLD:
            logger.warning(
                "CALIBRATION ALERT: Brier score %.3f exceeds threshold %.3f "
                "(n=%d games). Check for model drift.",
                brier, BRIER_ALERT_THRESHOLD, n,
            )
        else:
            logger.info(
                "Calibration metrics: method=%s, brier=%.3f, log_loss=%.3f, n=%d",
                self._method, brier, ll, n,
            )

        self.save()
        return {
            "method":      self._method,
            "n_games":     n,
            "brier_score": brier,
            "log_loss":    ll,
        }

    def _fit_platt(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression on raw probability)."""
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(solver="lbfgs", max_iter=500)
        lr.fit(raw_probs.reshape(-1, 1), actual_outcomes)
        self.platt_model = lr
        self._method     = "platt"
        self._is_fitted  = True
        logger.info(
            "Calibrator: Platt scaling fitted on %d games (coef=%.4f, intercept=%.4f).",
            len(raw_probs),
            float(lr.coef_[0][0]),
            float(lr.intercept_[0]),
        )

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to an array of raw probabilities.

        Parameters
        ----------
        raw_probs : np.ndarray or list of float — uncalibrated probabilities.

        Returns
        -------
        np.ndarray — calibrated probabilities clipped to [0.05, 0.95].
        """
        raw = np.asarray(raw_probs, dtype=float)

        if not self._is_fitted:
            return raw  # No calibration — return as-is

        if self._method == "isotonic" and self.isotonic_model is not None:
            calibrated = self.isotonic_model.predict(raw)
        elif self._method == "platt" and self.platt_model is not None:
            calibrated = self.platt_model.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            return raw

        return np.clip(calibrated, 0.05, 0.95)

    def calibrate_single(self, raw_prob: float) -> float:
        """Calibrate a single probability value."""
        return float(self.calibrate(np.array([raw_prob]))[0])

    def calibration_curve(
        self,
        raw_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Compute calibration curve (predicted probability vs. actual win rate by bucket).

        Returns
        -------
        dict with:
          bins         : list of (bin_lower, bin_upper) tuples
          mean_predicted: list of mean predicted probability per bin
          actual_rate  : list of actual win rate per bin
          counts       : list of sample count per bin
          max_deviation: float — worst-case deviation across all bins
        """
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins_out       = []
        mean_predicted = []
        actual_rate    = []
        counts         = []

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask   = (raw_probs >= lo) & (raw_probs < hi)
            if i == n_bins - 1:
                mask = (raw_probs >= lo) & (raw_probs <= hi)

            n_bin = mask.sum()
            if n_bin == 0:
                continue

            bins_out.append((round(lo, 2), round(hi, 2)))
            mean_predicted.append(round(float(np.mean(raw_probs[mask])), 3))
            actual_rate.append(round(float(np.mean(actual_outcomes[mask])), 3))
            counts.append(int(n_bin))

        max_dev = max(
            abs(p - a) for p, a in zip(mean_predicted, actual_rate)
        ) if mean_predicted else 0.0

        return {
            "bins":          bins_out,
            "mean_predicted": mean_predicted,
            "actual_rate":   actual_rate,
            "counts":        counts,
            "max_deviation": round(max_dev, 4),
        }

    def brier_score(
        self,
        raw_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        calibrated: bool = True,
    ) -> float:
        """
        Compute Brier score (lower is better; perfect = 0.0, coin flip = 0.25).

        Parameters
        ----------
        calibrated : If True, apply calibration before computing score.
        """
        from sklearn.metrics import brier_score_loss
        probs = self.calibrate(raw_probs) if calibrated else raw_probs
        return float(brier_score_loss(actual_outcomes, probs))

    def save(self) -> None:
        """Persist calibrator to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATOR_PATH, "wb") as f:
            pickle.dump({
                "platt_model":    self.platt_model,
                "isotonic_model": self.isotonic_model,
                "n_games":        self._n_games,
                "method":         self._method,
                "is_fitted":      self._is_fitted,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Calibrator saved to %s (method=%s, n=%d).",
                    CALIBRATOR_PATH, self._method, self._n_games)

    def _load_if_exists(self) -> None:
        if CALIBRATOR_PATH.exists():
            try:
                with open(CALIBRATOR_PATH, "rb") as f:
                    data = pickle.load(f)
                self.platt_model    = data.get("platt_model")
                self.isotonic_model = data.get("isotonic_model")
                self._n_games       = data.get("n_games", 0)
                self._method        = data.get("method", "none")
                self._is_fitted     = data.get("is_fitted", False)
                logger.info("Calibrator loaded from %s (method=%s, n=%d).",
                            CALIBRATOR_PATH, self._method, self._n_games)
            except Exception as exc:
                logger.warning("Could not load calibrator: %s", exc)

    def __repr__(self) -> str:
        return (
            f"ProbabilityCalibrator(method={self._method!r}, "
            f"n_games={self._n_games}, fitted={self._is_fitted})"
        )


def fit_from_db() -> dict:
    """
    Load all historical ensemble predictions + actual outcomes from DB
    and refit the calibrator. Called nightly after results are in.

    Returns
    -------
    dict — fit metrics from calibrator.fit().
    """
    try:
        with get_session() as session:
            rows = session.query(CalibrationHistory).filter(
                CalibrationHistory.actual_outcome.isnot(None)
            ).all()
    except Exception as exc:
        logger.error("DB query for calibration data failed: %s", exc)
        return {}

    if not rows:
        logger.info("No calibration history in DB yet — skipping calibrator refit.")
        return {}

    raw_probs       = np.array([r.ensemble_prob for r in rows], dtype=float)
    actual_outcomes = np.array([r.actual_outcome for r in rows], dtype=float)

    logger.info(
        "Refitting calibrator from DB: %d games, "
        "home win rate=%.3f, mean prediction=%.3f",
        len(rows), actual_outcomes.mean(), raw_probs.mean(),
    )

    calibrator = get_calibrator()
    return calibrator.fit(raw_probs, actual_outcomes)


def check_calibration_health(window_days: int = 30) -> dict:
    """
    Compute Brier score and calibration curve for the last `window_days` of predictions.
    Used for daily health monitoring. Returns health report dict.
    """
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=window_days)

    try:
        with get_session() as session:
            rows = session.query(CalibrationHistory).filter(
                CalibrationHistory.game_date >= cutoff.isoformat(),
                CalibrationHistory.actual_outcome.isnot(None),
            ).all()
    except Exception as exc:
        logger.error("Calibration health check DB query failed: %s", exc)
        return {"status": "error", "error": str(exc)}

    if len(rows) < 10:
        return {"status": "insufficient_data", "n_games": len(rows)}

    raw_probs       = np.array([r.ensemble_prob for r in rows], dtype=float)
    actual_outcomes = np.array([r.actual_outcome for r in rows], dtype=float)

    calibrator = get_calibrator()
    brier      = calibrator.brier_score(raw_probs, actual_outcomes, calibrated=True)
    brier_raw  = calibrator.brier_score(raw_probs, actual_outcomes, calibrated=False)
    curve      = calibrator.calibration_curve(raw_probs, actual_outcomes)
    accuracy   = float(np.mean((raw_probs >= 0.5) == actual_outcomes.astype(bool)))

    status = "GOOD" if brier < BRIER_ALERT_THRESHOLD else "ALERT"

    health = {
        "status":              status,
        "window_days":         window_days,
        "n_games":             len(rows),
        "brier_calibrated":    round(brier, 4),
        "brier_raw":           round(brier_raw, 4),
        "accuracy":            round(accuracy, 4),
        "calibration_method":  calibrator.method,
        "max_curve_deviation": curve["max_deviation"],
        "calibration_curve":   curve,
    }

    if status == "ALERT":
        logger.warning(
            "CALIBRATION HEALTH ALERT: Brier=%.3f (threshold=%.3f), "
            "n=%d games, window=%d days",
            brier, BRIER_ALERT_THRESHOLD, len(rows), window_days,
        )
    else:
        logger.info(
            "Calibration health OK: Brier=%.3f, accuracy=%.3f, n=%d games",
            brier, accuracy, len(rows),
        )

    return health


# Module-level singleton
_calibrator: Optional[ProbabilityCalibrator] = None


def get_calibrator() -> ProbabilityCalibrator:
    """Return the module-level calibrator singleton."""
    global _calibrator
    if _calibrator is None:
        _calibrator = ProbabilityCalibrator()
    return _calibrator


def calibrate(raw_prob: float) -> float:
    """Convenience function: calibrate a single probability."""
    return get_calibrator().calibrate_single(raw_prob)
