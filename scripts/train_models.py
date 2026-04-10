"""
Apex Analytics — Model Training Script
Trains the Random Forest and Logistic Regression ensemble layers on historical data.

Training strategy:
  - Training set:   2023 season (~2,430 games, post-shift-ban)
  - Holdout set:    2024 season (~2,430 games) — never seen during training
  - Hyperparameters: 5-fold cross-validation on training set only
  - Evaluation:     Accuracy, log loss, Brier score, AUC-ROC on 2024 holdout

Data requirement:
  Run  python -m scripts.build_historical_dataset  first to generate the CSVs.

Output:
  models/random_forest_model.pkl   — trained RF, ready for production
  models/logistic_model.pkl        — trained LR, ready for production
  data/historical/training_report.json — metrics and feature importance

Usage:
  python -m scripts.train_models
  python -m scripts.train_models --rf-only
  python -m scripts.train_models --lr-only
  python -m scripts.train_models --report-only   (load existing models, just evaluate)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature name constants (must match ensemble/random_forest_model.py
#                         and ensemble/logistic_model.py exactly)
# ---------------------------------------------------------------------------

RF_FEATURE_NAMES = [
    "home_starter_siera",
    "away_starter_siera",
    "home_starter_decay_xera",
    "away_starter_decay_xera",
    "home_team_xwoba",
    "away_team_xwoba",
    "home_bullpen_xfip",
    "away_bullpen_xfip",
    "park_factor_runs",
    "park_factor_hr",
    "home_decay_win_pct",
    "away_decay_win_pct",
    "weather_run_adj",
    "home_elo",
    "away_elo",
]

LR_FEATURE_NAMES = [
    "home_starter_era",
    "away_starter_era",
    "home_team_xwoba",
    "away_team_xwoba",
    "home_bullpen_xfip",
    "away_bullpen_xfip",
    "park_factor_runs",
    "home_decay_win_pct",
    "away_decay_win_pct",
    "weather_run_adj",
    "home_starter_decay_xera",
    "away_starter_decay_xera",
]

LABEL_COL = "home_win"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(year: int) -> Optional["pd.DataFrame"]:
    """Load the CSV dataset for a given season year."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed — run: pip install pandas")
        return None

    from config import HISTORICAL_DIR
    path = HISTORICAL_DIR / f"training_data_{year}.csv"

    if not path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "Run: python -m scripts.build_historical_dataset --year %d",
            path, year,
        )
        return None

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def prepare_matrices(
    df,
    feature_names: List[str],
    fill_na: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y from DataFrame."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        logger.warning("Missing features in dataset — filling with %.2f: %s", fill_na, missing)
        for c in missing:
            df[c] = fill_na

    X = df[feature_names].fillna(fill_na).values.astype(np.float64)
    y = df[LABEL_COL].values.astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Compute standard classification metrics for a trained model."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )
    except ImportError:
        logger.error("scikit-learn not installed — run: pip install scikit-learn")
        return {}

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)

    acc    = accuracy_score(y_test, preds)
    brier  = brier_score_loss(y_test, probs)
    ll     = log_loss(y_test, probs)
    auc    = roc_auc_score(y_test, probs)

    metrics = {
        "accuracy":    round(acc, 4),
        "brier_score": round(brier, 4),
        "log_loss":    round(ll, 4),
        "auc_roc":     round(auc, 4),
        "n_test":      len(y_test),
        "home_win_rate_test": round(y_test.mean(), 4),
    }

    logger.info("")
    logger.info("── %s Test-Set Metrics (2024 holdout) ──", model_name)
    logger.info("  Accuracy:    %.1f%%   (Vegas benchmark: ~58.2%%)", acc * 100)
    logger.info("  Brier Score: %.4f  (target: < 0.240)", brier)
    logger.info("  Log Loss:    %.4f", ll)
    logger.info("  AUC-ROC:     %.4f", auc)
    logger.info("  n games:     %d", len(y_test))

    return metrics


def calibration_summary(model, X_test: np.ndarray, y_test: np.ndarray, n_bins: int = 10) -> dict:
    """
    Bucket predictions into probability bins and compute actual win rate per bucket.
    Returns calibration curve data for reporting.
    """
    probs = model.predict_proba(X_test)[:, 1]

    bins        = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    actual_rates = []
    counts       = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (probs >= lo) & (probs < hi)
        n      = mask.sum()
        if n > 0:
            bin_centers.append(round(float((lo + hi) / 2), 3))
            actual_rates.append(round(float(y_test[mask].mean()), 4))
            counts.append(int(n))

    # Max calibration error
    if bin_centers:
        max_err = max(
            abs(pred - act)
            for pred, act in zip(bin_centers, actual_rates)
        )
    else:
        max_err = 0.0

    return {
        "bin_centers":   bin_centers,
        "actual_rates":  actual_rates,
        "counts":        counts,
        "max_error":     round(max_err, 4),
        "target_max_err": 0.05,
        "passes":        max_err <= 0.05,
    }


# ---------------------------------------------------------------------------
# Random Forest training
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> Tuple[object, dict]:
    """
    Train RandomForestClassifier with 5-fold GridSearchCV.
    Returns (fitted_model, metrics_dict).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        logger.error("scikit-learn not installed.")
        return None, {}

    logger.info("")
    logger.info("=" * 55)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("  Train: 2023 (%d games)  |  Test: 2024 (%d games)", len(y_train), len(y_test))
    logger.info("=" * 55)

    param_grid = {
        "n_estimators":     [200, 400],
        "max_depth":        [4, 6, 8],
        "min_samples_leaf": [20, 40, 80],
        "max_features":     ["sqrt", 0.5],
    }

    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Running 5-fold GridSearchCV over %d parameter combinations...",
                len(param_grid["n_estimators"]) *
                len(param_grid["max_depth"]) *
                len(param_grid["min_samples_leaf"]) *
                len(param_grid["max_features"]))

    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=5,
        scoring="neg_brier_score",   # Minimize Brier score
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logger.info("Best parameters: %s", best_params)
    logger.info("Best CV Brier: %.4f", -grid_search.best_score_)

    # Feature importance
    importances = dict(zip(RF_FEATURE_NAMES, best_model.feature_importances_))
    sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("")
    logger.info("Feature Importances (Random Forest):")
    for fname, imp in sorted_imp:
        bar = "█" * int(imp * 40)
        logger.info("  %-30s %.4f  %s", fname, imp, bar)

    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test, "Random Forest")
    metrics["best_params"]         = best_params
    metrics["cv_brier_score"]      = round(-grid_search.best_score_, 4)
    metrics["feature_importances"] = {k: round(v, 6) for k, v in sorted_imp}

    # Calibration check
    cal = calibration_summary(best_model, X_test, y_test)
    metrics["calibration"] = cal
    logger.info(
        "Calibration max error: %.3f  (target: ≤ 0.05) — %s",
        cal["max_error"],
        "PASS ✓" if cal["passes"] else "NEEDS CALIBRATION ⚠",
    )

    return best_model, metrics


# ---------------------------------------------------------------------------
# Logistic Regression training
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> Tuple[object, dict]:
    """
    Train LogisticRegression with L2 regularization and GridSearchCV for C.
    Uses StandardScaler (fitted on training data only).
    Returns (fitted_pipeline, metrics_dict).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn not installed.")
        return None, {}

    logger.info("")
    logger.info("=" * 55)
    logger.info("TRAINING LOGISTIC REGRESSION")
    logger.info("  Train: 2023 (%d games)  |  Test: 2024 (%d games)", len(y_train), len(y_test))
    logger.info("=" * 55)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    param_grid = {"lr__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}

    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="neg_brier_score",
        refit=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_C        = grid_search.best_params_["lr__C"]
    logger.info("Best C: %s  |  CV Brier: %.4f", best_C, -grid_search.best_score_)

    # Named coefficients
    lr_step  = best_pipeline.named_steps["lr"]
    coefs    = dict(zip(LR_FEATURE_NAMES, lr_step.coef_[0]))
    sorted_c = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("")
    logger.info("Coefficients (Logistic Regression, sorted by |coef|):")
    for fname, coef in sorted_c:
        direction = "→ favors home" if coef < 0 else "→ favors away"
        logger.info("  %-30s %+.4f  %s", fname, coef, direction)

    # Evaluate
    metrics = evaluate_model(best_pipeline, X_test, y_test, "Logistic Regression")
    metrics["best_C"]        = best_C
    metrics["cv_brier_score"] = round(-grid_search.best_score_, 4)
    metrics["coefficients"]  = {k: round(v, 6) for k, v in sorted_c}

    # Calibration
    cal = calibration_summary(best_pipeline, X_test, y_test)
    metrics["calibration"] = cal
    logger.info(
        "Calibration max error: %.3f  (target: ≤ 0.05) — %s",
        cal["max_error"],
        "PASS ✓" if cal["passes"] else "NEEDS CALIBRATION ⚠",
    )

    return best_pipeline, metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_rf_model(model, metrics: dict) -> None:
    """Save trained RF to models/random_forest_model.pkl via the ensemble class."""
    try:
        from ensemble.random_forest_model import RandomForestModel
        wrapper = RandomForestModel.__new__(RandomForestModel)
        wrapper.model                = model
        wrapper.feature_importances_ = metrics.get("feature_importances", {})
        wrapper._is_trained          = True
        wrapper.save()
        logger.info("Random Forest saved → models/random_forest_model.pkl")
    except Exception as exc:
        logger.error("Failed to save RF via wrapper: %s", exc)
        # Direct save fallback
        import pickle
        from config import MODELS_DIR
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / "random_forest_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "feature_names": RF_FEATURE_NAMES}, f)
        logger.info("RF saved directly → %s", path)


def save_lr_model(model, metrics: dict) -> None:
    """Save trained LR pipeline to models/logistic_model.pkl via ensemble class."""
    try:
        from ensemble.logistic_model import LogisticModel
        wrapper = LogisticModel.__new__(LogisticModel)
        wrapper.model        = model
        wrapper._is_trained  = True
        wrapper.scaler       = model.named_steps.get("scaler")
        wrapper.coefficients_ = metrics.get("coefficients", {})
        wrapper.save()
        logger.info("Logistic Regression saved → models/logistic_model.pkl")
    except Exception as exc:
        logger.error("Failed to save LR via wrapper: %s", exc)
        import pickle
        from config import MODELS_DIR
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / "logistic_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "feature_names": LR_FEATURE_NAMES}, f)
        logger.info("LR saved directly → %s", path)


def save_training_report(rf_metrics: dict, lr_metrics: dict) -> None:
    """Save combined training metrics to JSON for logging and review."""
    from config import HISTORICAL_DIR
    import datetime

    report = {
        "generated_at":      datetime.datetime.now().isoformat(),
        "training_year":     2023,
        "test_year":         2024,
        "random_forest":     rf_metrics,
        "logistic_regression": lr_metrics,
        "comparison": {
            "rf_accuracy":     rf_metrics.get("accuracy"),
            "lr_accuracy":     lr_metrics.get("accuracy"),
            "rf_brier":        rf_metrics.get("brier_score"),
            "lr_brier":        lr_metrics.get("brier_score"),
            "winner_accuracy": (
                "RandomForest" if (rf_metrics.get("accuracy", 0) > lr_metrics.get("accuracy", 0))
                else "LogisticRegression"
            ),
            "winner_brier": (
                "RandomForest" if (rf_metrics.get("brier_score", 1) < lr_metrics.get("brier_score", 1))
                else "LogisticRegression"
            ),
        },
    }

    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HISTORICAL_DIR / "training_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Training report saved → %s", out_path)

    # Print summary comparison
    logger.info("")
    logger.info("=" * 55)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 55)
    logger.info("                    RF          LR")
    logger.info("Accuracy:       %5.1f%%       %5.1f%%       (Vegas ~58.2%%)",
                rf_metrics.get("accuracy", 0) * 100,
                lr_metrics.get("accuracy", 0) * 100)
    logger.info("Brier Score:    %6.4f      %6.4f      (target < 0.240)",
                rf_metrics.get("brier_score", 0),
                lr_metrics.get("brier_score", 0))
    logger.info("Log Loss:       %6.4f      %6.4f",
                rf_metrics.get("log_loss", 0),
                lr_metrics.get("log_loss", 0))
    logger.info("AUC-ROC:        %6.4f      %6.4f",
                rf_metrics.get("auc_roc", 0),
                lr_metrics.get("auc_roc", 0))
    logger.info("")
    logger.info("Both models saved and ready for production ensemble.")


# ---------------------------------------------------------------------------
# Report-only mode (load existing models, evaluate on test data)
# ---------------------------------------------------------------------------

def evaluate_existing_models() -> None:
    """Load saved models and re-evaluate on 2024 holdout data."""
    logger.info("Report-only mode: loading existing saved models...")

    df_test = load_dataset(2024)
    if df_test is None:
        return

    X_rf, y = prepare_matrices(df_test, RF_FEATURE_NAMES)
    X_lr, _ = prepare_matrices(df_test, LR_FEATURE_NAMES)

    try:
        from ensemble.random_forest_model import get_rf_model
        rf = get_rf_model()
        if rf._is_trained:
            evaluate_model(rf.model, X_rf, y, "Random Forest (loaded)")
            cal = calibration_summary(rf.model, X_rf, y)
            logger.info("RF calibration max error: %.4f", cal["max_error"])
        else:
            logger.warning("RF model not trained — no evaluation.")
    except Exception as exc:
        logger.warning("Could not load RF: %s", exc)

    try:
        from ensemble.logistic_model import get_lr_model
        lr = get_lr_model()
        if lr._is_trained:
            evaluate_model(lr.model, X_lr, y, "Logistic Regression (loaded)")
            cal = calibration_summary(lr.model, X_lr, y)
            logger.info("LR calibration max error: %.4f", cal["max_error"])
        else:
            logger.warning("LR model not trained — no evaluation.")
    except Exception as exc:
        logger.warning("Could not load LR: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    train_rf:    bool = True,
    train_lr:    bool = True,
    report_only: bool = False,
) -> None:
    """Train RF and LR on 2023 season, evaluate on 2024 holdout."""

    if report_only:
        evaluate_existing_models()
        return

    # Load datasets
    df_train = load_dataset(2023)
    df_test  = load_dataset(2024)

    if df_train is None or df_test is None:
        logger.error(
            "Cannot train — datasets missing.\n"
            "Run: python -m scripts.build_historical_dataset"
        )
        sys.exit(1)

    rf_metrics: dict = {}
    lr_metrics: dict = {}

    if train_rf:
        X_train_rf, y_train = prepare_matrices(df_train, RF_FEATURE_NAMES)
        X_test_rf,  y_test  = prepare_matrices(df_test,  RF_FEATURE_NAMES)
        rf_model, rf_metrics = train_random_forest(X_train_rf, y_train, X_test_rf, y_test)
        if rf_model:
            save_rf_model(rf_model, rf_metrics)

    if train_lr:
        X_train_lr, y_train = prepare_matrices(df_train, LR_FEATURE_NAMES)
        X_test_lr,  y_test  = prepare_matrices(df_test,  LR_FEATURE_NAMES)
        lr_model, lr_metrics = train_logistic_regression(X_train_lr, y_train, X_test_lr, y_test)
        if lr_model:
            save_lr_model(lr_model, lr_metrics)

    if rf_metrics and lr_metrics:
        save_training_report(rf_metrics, lr_metrics)
    elif rf_metrics:
        logger.info("RF training complete. Brier=%.4f  Acc=%.1f%%",
                    rf_metrics.get("brier_score", 0), rf_metrics.get("accuracy", 0) * 100)
    elif lr_metrics:
        logger.info("LR training complete. Brier=%.4f  Acc=%.1f%%",
                    lr_metrics.get("brier_score", 0), lr_metrics.get("accuracy", 0) * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Apex Analytics ensemble models")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--rf-only",      action="store_true", help="Train Random Forest only")
    group.add_argument("--lr-only",      action="store_true", help="Train Logistic Regression only")
    group.add_argument("--report-only",  action="store_true", help="Evaluate existing saved models")
    args = parser.parse_args()

    main(
        train_rf    = not args.lr_only,
        train_lr    = not args.rf_only,
        report_only = args.report_only,
    )
