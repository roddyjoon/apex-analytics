"""
Apex Analytics — Random Forest Ensemble Layer
Non-linear feature interactions that logistic regression cannot capture.

Key insight: a great pitcher matters MORE when the opposing offense is weak
(interaction effect), and park factor matters LESS in cold weather
(conditional relationship). Linear models miss these entirely.

Training: 2023–2024 only (post-shift-ban, modern player profiles).
Features: 15 — RF handles collinearity natively; more features are fine.
Output: Win probability (0.0–1.0) for the home team.

Model persistence: models/random_forest_model.pkl
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
FEATURE_NAMES = [
    "home_starter_siera",           # 1. Home SP true SIERA
    "away_starter_siera",           # 2. Away SP true SIERA
    "home_starter_decay_xera",      # 3. Home SP exp-decay xERA (last 5 starts)
    "away_starter_decay_xera",      # 4. Away SP exp-decay xERA (last 5 starts)
    "home_team_xwoba",              # 5. Home lineup season avg xwOBA (park-adj)
    "away_team_xwoba",              # 6. Away lineup season avg xwOBA (park-adj)
    "home_bullpen_xfip",            # 7. Home bullpen xFIP
    "away_bullpen_xfip",            # 8. Away bullpen xFIP
    "park_factor_runs",             # 9. Run park factor
    "park_factor_hr",               # 10. HR park factor
    "home_decay_win_pct",           # 11. Home team exp-decay win% (last 15 games)
    "away_decay_win_pct",           # 12. Away team exp-decay win% (last 15 games)
    "weather_run_adj",              # 13. Net weather run adjustment
    "home_elo",                     # 14. Home team Elo rating
    "away_elo",                     # 15. Away team Elo rating
]


class RandomForestModel:
    """
    Random Forest wrapper for win probability prediction.
    Loads a pre-trained model or uses a calibrated baseline until trained.
    """

    def __init__(self):
        self.model = None
        self.feature_importances_: Optional[dict] = None
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
            prob = self.model.predict_proba(X.reshape(1, -1))[0][1]
            return float(np.clip(prob, 0.05, 0.95))
        except Exception as exc:
            logger.warning("RF prediction failed: %s — using baseline.", exc)
            return self._baseline_prediction(features)

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the Random Forest on historical game data.

        Parameters
        ----------
        X : Feature matrix (n_games × 15 features).
        y : Labels (1 = home team won, 0 = home team lost).

        Returns
        -------
        dict with training metrics: accuracy, log_loss, auc_roc, brier_score.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, GridSearchCV
        from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

        logger.info("Training Random Forest on %d games, %d features...", X.shape[0], X.shape[1])

        # Hyperparameter search (fast grid for 5,000-game dataset)
        param_grid = {
            "n_estimators":    [200, 400],
            "max_depth":       [6, 10, None],
            "min_samples_leaf": [20, 50],
        }
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
        grid    = GridSearchCV(base_rf, param_grid, cv=5, scoring="neg_log_loss",
                               n_jobs=-1, verbose=0)
        grid.fit(X, y)

        self.model = grid.best_estimator_
        self._is_trained = True

        # Feature importance
        self.feature_importances_ = dict(
            zip(FEATURE_NAMES, self.model.feature_importances_)
        )

        # Evaluate on training data (holdout eval done in scripts/train_models.py)
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "best_params":   grid.best_params_,
            "train_accuracy": float(np.mean(y_pred == y)),
            "train_log_loss": float(log_loss(y, y_prob)),
            "train_auc_roc":  float(roc_auc_score(y, y_prob)),
            "train_brier":    float(brier_score_loss(y, y_prob)),
            "n_samples":      len(y),
            "feature_importances": self.feature_importances_,
        }

        logger.info(
            "RF trained: accuracy=%.3f, log_loss=%.3f, AUC=%.3f, brier=%.3f | best=%s",
            metrics["train_accuracy"], metrics["train_log_loss"],
            metrics["train_auc_roc"],  metrics["train_brier"],
            grid.best_params_
        )
        self.save()
        return metrics

    def save(self) -> None:
        """Persist trained model to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_importances": self.feature_importances_,
                "is_trained": self._is_trained,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("RF model saved to %s", MODEL_PATH)

    def _load_if_exists(self) -> None:
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self.model                = data["model"]
                self.feature_importances_ = data.get("feature_importances")
                self._is_trained          = data.get("is_trained", False)
                logger.info("RF model loaded from %s", MODEL_PATH)
            except Exception as exc:
                logger.warning("Could not load RF model: %s", exc)
                self._is_trained = False

    def get_feature_importance(self) -> dict:
        """Return feature importances sorted descending."""
        if not self.feature_importances_:
            return {}
        return dict(sorted(self.feature_importances_.items(),
                           key=lambda x: x[1], reverse=True))

    @staticmethod
    def build_features(
        home_starter_siera:      float,
        away_starter_siera:      float,
        home_starter_decay_xera: float,
        away_starter_decay_xera: float,
        home_team_xwoba:         float,
        away_team_xwoba:         float,
        home_bullpen_xfip:       float,
        away_bullpen_xfip:       float,
        park_factor_runs:        float,
        park_factor_hr:          float,
        home_decay_win_pct:      float,
        away_decay_win_pct:      float,
        weather_run_adj:         float,
        home_elo:                float,
        away_elo:                float,
    ) -> dict:
        """Build a feature dict from named arguments. All 15 features required."""
        return {
            "home_starter_siera":       home_starter_siera,
            "away_starter_siera":       away_starter_siera,
            "home_starter_decay_xera":  home_starter_decay_xera,
            "away_starter_decay_xera":  away_starter_decay_xera,
            "home_team_xwoba":          home_team_xwoba,
            "away_team_xwoba":          away_team_xwoba,
            "home_bullpen_xfip":        home_bullpen_xfip,
            "away_bullpen_xfip":        away_bullpen_xfip,
            "park_factor_runs":         park_factor_runs,
            "park_factor_hr":           park_factor_hr,
            "home_decay_win_pct":       home_decay_win_pct,
            "away_decay_win_pct":       away_decay_win_pct,
            "weather_run_adj":          weather_run_adj,
            "home_elo":                 home_elo,
            "away_elo":                 away_elo,
        }

    def _build_feature_vector(self, features: dict) -> np.ndarray:
        """Convert feature dict to numpy array in the correct column order."""
        defaults = _feature_defaults()
        return np.array([features.get(name, defaults[name]) for name in FEATURE_NAMES],
                        dtype=float)

    def _baseline_prediction(self, features: dict) -> float:
        """
        Heuristic prediction when no trained model exists.
        Uses SIERA and Elo differentials as simple signal.
        """
        home_siera = features.get("home_starter_siera", 4.20)
        away_siera = features.get("away_starter_siera", 4.20)
        home_elo   = features.get("home_elo", 1500.0)
        away_elo   = features.get("away_elo", 1500.0)

        # SIERA edge: better pitcher → higher win probability
        siera_edge = (away_siera - home_siera) / 8.0  # Normalize to ~0.0-0.15 range

        # Elo edge
        from ensemble.elo_system import win_probability
        elo_prob = win_probability(home_elo, away_elo)

        # Blend: 60% Elo, 40% SIERA signal
        raw = (elo_prob * 0.60) + ((0.50 + siera_edge) * 0.40)
        return float(np.clip(raw, 0.10, 0.90))


def _feature_defaults() -> dict:
    """Default feature values (league averages) for missing data."""
    return {
        "home_starter_siera":       4.20,
        "away_starter_siera":       4.20,
        "home_starter_decay_xera":  4.20,
        "away_starter_decay_xera":  4.20,
        "home_team_xwoba":          0.320,
        "away_team_xwoba":          0.320,
        "home_bullpen_xfip":        4.20,
        "away_bullpen_xfip":        4.20,
        "park_factor_runs":         1.00,
        "park_factor_hr":           1.00,
        "home_decay_win_pct":       0.500,
        "away_decay_win_pct":       0.500,
        "weather_run_adj":          0.00,
        "home_elo":                 1500.0,
        "away_elo":                 1500.0,
    }


# Module-level singleton (loaded once at import time)
_rf_model: Optional[RandomForestModel] = None


def get_rf_model() -> RandomForestModel:
    """Return the module-level RF model singleton."""
    global _rf_model
    if _rf_model is None:
        _rf_model = RandomForestModel()
    return _rf_model


def predict(features: dict) -> float:
    """Convenience function: predict home win probability from feature dict."""
    return get_rf_model().predict_win_probability(features)
