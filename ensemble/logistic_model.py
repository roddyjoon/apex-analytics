"""
Apex Analytics — Logistic Regression Ensemble Layer
Interpretable baseline: linear model with 12 features, L2 regularization.

Why logistic regression:
  - Human-readable coefficients show exactly what the model weights
  - Strong regularization generalizes well in April (small sample season)
  - Fast, stable, no hyperparameter explosion
  - Calibration is well-behaved (sigmoid output is nearly calibrated)
  - Cross-validates well on 5,000-game dataset

Features (12 — kept lean for interpretability + strong L2 regularization):
  1.  Home starter true-talent ERA (SIERA blend)
  2.  Away starter true-talent ERA (SIERA blend)
  3.  Home team xwOBA (season avg, park-adjusted)
  4.  Away team xwOBA (season avg, park-adjusted)
  5.  Home bullpen xFIP
  6.  Away bullpen xFIP
  7.  Park factor (runs)
  8.  Home team exp-decay win% (last 15 games)
  9.  Away team exp-decay win% (last 15 games)
  10. Weather run adjustment
  11. Home starter exp-decay xERA (last 5 starts)
  12. Away starter exp-decay xERA (last 5 starts)

Training: 2023–2024 only (post-shift-ban, modern player profiles).
Output: Win probability (0.0–1.0) for the home team.

Model persistence: models/logistic_model.pkl
"""

import logging
import pickle
from typing import Optional

import numpy as np

from config import MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "logistic_model.pkl"

FEATURE_NAMES = [
    "home_starter_era",        # 1. Home SP true-talent ERA (SIERA blend)
    "away_starter_era",        # 2. Away SP true-talent ERA (SIERA blend)
    "home_team_xwoba",         # 3. Home lineup season avg xwOBA (park-adj)
    "away_team_xwoba",         # 4. Away lineup season avg xwOBA (park-adj)
    "home_bullpen_xfip",       # 5. Home bullpen xFIP
    "away_bullpen_xfip",       # 6. Away bullpen xFIP
    "park_factor_runs",        # 7. Run park factor
    "home_decay_win_pct",      # 8. Home team exp-decay win% (last 15 games)
    "away_decay_win_pct",      # 9. Away team exp-decay win% (last 15 games)
    "weather_run_adj",         # 10. Net weather run adjustment
    "home_starter_decay_xera", # 11. Home SP exp-decay xERA (last 5 starts)
    "away_starter_decay_xera", # 12. Away SP exp-decay xERA (last 5 starts)
]


class LogisticModel:
    """
    Logistic Regression wrapper for win probability prediction.
    Loads a pre-trained model or uses a calibrated baseline until trained.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.coefficients_: Optional[dict] = None
        self._is_trained = False
        self._load_if_exists()

    def predict_win_probability(self, features: dict) -> float:
        """
        Predict home team win probability.

        Parameters
        ----------
        features : Dict with keys matching FEATURE_NAMES.
                   Missing features are filled with league-average defaults.

        Returns
        -------
        float — home win probability (0.0–1.0).
        """
        if not self._is_trained or self.model is None:
            return self._baseline_prediction(features)

        X = self._build_feature_vector(features)
        try:
            # If model is a Pipeline (contains its own scaler), pass raw features directly.
            # Otherwise apply standalone scaler before predicting.
            from sklearn.pipeline import Pipeline as _Pipeline
            if isinstance(self.model, _Pipeline):
                X_input = X.reshape(1, -1)
            elif self.scaler is not None:
                X_input = self.scaler.transform(X.reshape(1, -1))
            else:
                X_input = X.reshape(1, -1)
            prob = self.model.predict_proba(X_input)[0][1]
            return float(np.clip(prob, 0.05, 0.95))
        except Exception as exc:
            logger.warning("LR prediction failed: %s — using baseline.", exc)
            return self._baseline_prediction(features)

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train Logistic Regression on historical game data.

        Parameters
        ----------
        X : Feature matrix (n_games × 12 features).
        y : Labels (1 = home team won, 0 = home team lost).

        Returns
        -------
        dict with training metrics: accuracy, log_loss, auc_roc, brier_score, coefficients.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
        from sklearn.pipeline import Pipeline

        logger.info("Training Logistic Regression on %d games, %d features...",
                    X.shape[0], X.shape[1])

        # Feature scaling is critical for LR (L2 regularization is scale-sensitive)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Hyperparameter search: tune regularization strength C
        # Small C = stronger regularization (better for sparse April data)
        param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}
        base_lr = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )
        grid = GridSearchCV(
            base_lr, param_grid, cv=5,
            scoring="neg_log_loss",
            n_jobs=-1, verbose=0,
        )
        grid.fit(X_scaled, y)

        self.model = grid.best_estimator_
        self._is_trained = True

        # Extract named coefficients (interpretability is the main LR advantage)
        self.coefficients_ = dict(
            zip(FEATURE_NAMES, self.model.coef_[0])
        )

        # Evaluation on training data
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "best_C":          grid.best_params_["C"],
            "train_accuracy":  float(np.mean(y_pred == y)),
            "train_log_loss":  float(log_loss(y, y_prob)),
            "train_auc_roc":   float(roc_auc_score(y, y_prob)),
            "train_brier":     float(brier_score_loss(y, y_prob)),
            "n_samples":       len(y),
            "coefficients":    self.coefficients_,
        }

        logger.info(
            "LR trained: accuracy=%.3f, log_loss=%.3f, AUC=%.3f, brier=%.3f | C=%.3f",
            metrics["train_accuracy"], metrics["train_log_loss"],
            metrics["train_auc_roc"],  metrics["train_brier"],
            grid.best_params_["C"],
        )
        self._log_coefficients()
        self.save()
        return metrics

    def save(self) -> None:
        """Persist trained model and scaler to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":        self.model,
                "scaler":       self.scaler,
                "coefficients": self.coefficients_,
                "is_trained":   self._is_trained,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LR model saved to %s", MODEL_PATH)

    def _load_if_exists(self) -> None:
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self.model          = data["model"]
                self.scaler         = data.get("scaler")
                self.coefficients_  = data.get("coefficients")
                self._is_trained    = data.get("is_trained", False)
                logger.info("LR model loaded from %s", MODEL_PATH)
            except Exception as exc:
                logger.warning("Could not load LR model: %s", exc)
                self._is_trained = False

    def get_coefficients(self) -> dict:
        """Return feature coefficients sorted by absolute value (descending)."""
        if not self.coefficients_:
            return {}
        return dict(sorted(
            self.coefficients_.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ))

    def _log_coefficients(self) -> None:
        """Log top coefficients for interpretability monitoring."""
        if not self.coefficients_:
            return
        ranked = self.get_coefficients()
        logger.info("LR top coefficients (absolute value):")
        for feat, coef in list(ranked.items())[:6]:
            direction = "↑ home" if coef > 0 else "↓ home"
            logger.info("  %s: %.4f (%s)", feat, coef, direction)

    @staticmethod
    def build_features(
        home_starter_era:        float,
        away_starter_era:        float,
        home_team_xwoba:         float,
        away_team_xwoba:         float,
        home_bullpen_xfip:       float,
        away_bullpen_xfip:       float,
        park_factor_runs:        float,
        home_decay_win_pct:      float,
        away_decay_win_pct:      float,
        weather_run_adj:         float,
        home_starter_decay_xera: float,
        away_starter_decay_xera: float,
    ) -> dict:
        """Build a feature dict from named arguments. All 12 features required."""
        return {
            "home_starter_era":        home_starter_era,
            "away_starter_era":        away_starter_era,
            "home_team_xwoba":         home_team_xwoba,
            "away_team_xwoba":         away_team_xwoba,
            "home_bullpen_xfip":       home_bullpen_xfip,
            "away_bullpen_xfip":       away_bullpen_xfip,
            "park_factor_runs":        park_factor_runs,
            "home_decay_win_pct":      home_decay_win_pct,
            "away_decay_win_pct":      away_decay_win_pct,
            "weather_run_adj":         weather_run_adj,
            "home_starter_decay_xera": home_starter_decay_xera,
            "away_starter_decay_xera": away_starter_decay_xera,
        }

    def _build_feature_vector(self, features: dict) -> np.ndarray:
        """Convert feature dict to numpy array in the correct column order."""
        defaults = _feature_defaults()
        return np.array(
            [features.get(name, defaults[name]) for name in FEATURE_NAMES],
            dtype=float,
        )

    def _baseline_prediction(self, features: dict) -> float:
        """
        Heuristic prediction when no trained model exists.
        Uses ERA differential and recent win% as signal.
        """
        home_era     = features.get("home_starter_era", 4.20)
        away_era     = features.get("away_starter_era", 4.20)
        home_win_pct = features.get("home_decay_win_pct", 0.500)
        away_win_pct = features.get("away_decay_win_pct", 0.500)
        home_xwoba   = features.get("home_team_xwoba", 0.320)
        away_xwoba   = features.get("away_team_xwoba", 0.320)

        # ERA edge: home pitcher advantage (lower ERA = better)
        era_edge   = (away_era - home_era) / 8.0

        # Offense edge
        xwoba_edge = (home_xwoba - away_xwoba) / 0.08

        # Recent form edge
        form_edge  = (home_win_pct - away_win_pct) * 0.5

        # Blend into home win probability
        raw = 0.53 + (era_edge * 0.35) + (xwoba_edge * 0.25) + (form_edge * 0.10)
        return float(np.clip(raw, 0.10, 0.90))


def _feature_defaults() -> dict:
    """Default feature values (league averages) for missing data."""
    return {
        "home_starter_era":        4.20,
        "away_starter_era":        4.20,
        "home_team_xwoba":         0.320,
        "away_team_xwoba":         0.320,
        "home_bullpen_xfip":       4.20,
        "away_bullpen_xfip":       4.20,
        "park_factor_runs":        1.00,
        "home_decay_win_pct":      0.500,
        "away_decay_win_pct":      0.500,
        "weather_run_adj":         0.00,
        "home_starter_decay_xera": 4.20,
        "away_starter_decay_xera": 4.20,
    }


# Module-level singleton (loaded once at import time)
_lr_model: Optional[LogisticModel] = None


def get_lr_model() -> LogisticModel:
    """Return the module-level LR model singleton."""
    global _lr_model
    if _lr_model is None:
        _lr_model = LogisticModel()
    return _lr_model


def predict(features: dict) -> float:
    """Convenience function: predict home win probability from feature dict."""
    return get_lr_model().predict_win_probability(features)
