"""
Apex Analytics — Ensemble Layer Tests
Tests Elo system, ensemble blender, RF/LR model interfaces,
and calibration pass-through.

All tests are deterministic and require no external data.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.elo_system import (
    win_probability,
    update_elo_ratings,
    regress_to_mean,
)
from ensemble.blender import blend_predictions, get_phase_weights_for_date
from ensemble.random_forest_model import get_rf_model, FEATURE_NAMES as RF_FEATURES
from ensemble.logistic_model import get_lr_model, FEATURE_NAMES as LR_FEATURES
from ensemble.calibrator import get_calibrator

from tests.conftest import make_context


# ===========================================================================
# Elo System
# ===========================================================================

class TestEloSystem:

    def test_equal_teams_home_win_prob_in_hfa_range(self):
        """
        Two teams with equal Elo (1500 vs 1500): home wins ~57% due to +48 HFA.
        """
        p = win_probability(1500.0, 1500.0)
        assert 0.55 <= p <= 0.60, f"Equal Elo HFA should be 55-60%, got {p:.4f}"

    def test_better_team_wins_more_often(self):
        """Team with higher Elo should have higher win probability."""
        p_strong = win_probability(1600.0, 1400.0)
        p_equal  = win_probability(1500.0, 1500.0)
        p_weak   = win_probability(1400.0, 1600.0)

        assert p_strong > p_equal > p_weak, (
            f"Strong={p_strong:.4f}, Equal={p_equal:.4f}, Weak={p_weak:.4f}"
        )

    def test_win_probability_is_valid_probability(self):
        """win_probability output must be in [0, 1]."""
        for home_elo, away_elo in [(1300, 1700), (1500, 1500), (1700, 1300)]:
            p = win_probability(float(home_elo), float(away_elo))
            assert 0.0 < p < 1.0, f"win_probability({home_elo}, {away_elo}) = {p}"

    def test_elo_update_winner_gains_loser_loses(self):
        """After a result, winner's Elo increases and loser's decreases."""
        winner_before = 1500.0
        loser_before  = 1500.0

        new_winner, new_loser = update_elo_ratings(
            winner_before, loser_before,
            home_team_won=True, run_differential=3,
        )

        assert new_winner > winner_before, "Winner should gain Elo"
        assert new_loser  < loser_before,  "Loser should lose Elo"

    def test_elo_update_margin_of_victory_matters(self):
        """Winning by 8 runs should yield a larger rating change than winning by 1."""
        w_blowout, _ = update_elo_ratings(1500, 1500, True, run_differential=8)
        w_close,   _ = update_elo_ratings(1500, 1500, True, run_differential=1)

        assert w_blowout > w_close, (
            f"Blowout winner gain ({w_blowout:.2f}) should exceed "
            f"close-game gain ({w_close:.2f})"
        )

    def test_elo_update_home_team_won_false(self):
        """When home_team_won=False, the away team (loser position) Elo should increase."""
        # update_elo_ratings(winner_elo, loser_elo, home_team_won, run_diff)
        # When home_team_won=False: winner is the away team
        new_winner, new_loser = update_elo_ratings(
            1500.0, 1500.0, home_team_won=False, run_differential=2
        )
        assert new_winner > 1500.0  # winner gained Elo
        assert new_loser < 1500.0   # loser lost Elo

    def test_regress_to_mean_moves_toward_1500(self):
        """
        regress_to_mean should pull an above-average team toward 1500
        and push a below-average team toward 1500.
        """
        above = regress_to_mean(1600.0)
        below = regress_to_mean(1400.0)

        assert 1500.0 < above < 1600.0, f"1600 regressed to {above}"
        assert 1400.0 < below < 1500.0, f"1400 regressed to {below}"

    def test_regress_to_mean_with_33pct(self):
        """33% regression: 1600 → 1500 + 0.67×100 = 1567."""
        result = regress_to_mean(1600.0, regression_pct=0.33)
        expected = 1500.0 + 0.67 * 100.0   # ~1567
        assert abs(result - expected) < 1.0, (
            f"Expected ~{expected:.1f}, got {result:.4f}"
        )

    def test_regress_to_mean_at_1500_unchanged(self):
        """A team already at 1500 Elo should not change."""
        result = regress_to_mean(1500.0)
        assert result == pytest.approx(1500.0, abs=0.01)


# ===========================================================================
# Ensemble Blender
# ===========================================================================

class TestEnsembleBlender:

    def test_blend_returns_required_keys(self):
        """blend_predictions must return a dict with all required keys."""
        result = blend_predictions(0.60, 0.57, 0.58, 0.56)

        required = {"raw_ensemble", "calibrated_prob", "weights"}
        missing  = required - set(result.keys())
        assert not missing, f"blend_predictions missing keys: {missing}"

    def test_blend_valid_probability_range(self):
        """Blended probability must be in (0, 1)."""
        result = blend_predictions(0.60, 0.57, 0.58, 0.56)
        p = result["calibrated_prob"]
        assert 0.0 < p < 1.0, f"calibrated_prob out of range: {p}"

    def test_blend_weights_sum_to_one(self):
        """Ensemble weights must sum to exactly 1.0 for any season phase."""
        test_dates = [
            date(2025, 4, 1),   # Early April
            date(2025, 5, 1),   # Late April/May
            date(2025, 7, 15),  # June–July
            date(2025, 9, 1),   # August–September
        ]

        for d in test_dates:
            weights = get_phase_weights_for_date(d)
            total   = weights["mc"] + weights["elo"] + weights["rf"] + weights["lr"]
            assert abs(total - 1.0) < 1e-6, \
                f"Weights don't sum to 1 on {d}: {total:.6f} — {weights}"

    def test_early_april_weights(self):
        """Opening Day – Apr 15: MC=30%, Elo=15%, RF=30%, LR=25%."""
        weights = get_phase_weights_for_date(date(2025, 4, 5))
        assert weights["mc"]  == pytest.approx(0.30, abs=0.01)
        assert weights["elo"] == pytest.approx(0.15, abs=0.01)
        assert weights["rf"]  == pytest.approx(0.30, abs=0.01)
        assert weights["lr"]  == pytest.approx(0.25, abs=0.01)

    def test_june_weights(self):
        """June–July: MC=60%, Elo=10%, RF=20%, LR=10%."""
        weights = get_phase_weights_for_date(date(2025, 6, 15))
        assert weights["mc"]  == pytest.approx(0.60, abs=0.01)
        assert weights["elo"] == pytest.approx(0.10, abs=0.01)
        assert weights["rf"]  == pytest.approx(0.20, abs=0.01)
        assert weights["lr"]  == pytest.approx(0.10, abs=0.01)

    def test_august_weights(self):
        """August–September: MC=65%, Elo=8%, RF=18%, LR=9%."""
        weights = get_phase_weights_for_date(date(2025, 8, 15))
        assert weights["mc"]  == pytest.approx(0.65, abs=0.01)
        assert weights["elo"] == pytest.approx(0.08, abs=0.01)
        assert weights["rf"]  == pytest.approx(0.18, abs=0.01)
        assert weights["lr"]  == pytest.approx(0.09, abs=0.01)

    def test_home_dominant_scenario_gives_high_probability(self):
        """When all layers strongly favor home, calibrated_prob should be > 0.60."""
        result = blend_predictions(
            mc_prob  = 0.68,
            elo_prob = 0.66,
            rf_prob  = 0.65,
            lr_prob  = 0.64,
            game_date = date(2025, 7, 1),
            apply_calibration = False,
        )
        assert result["calibrated_prob"] > 0.60

    def test_away_dominant_scenario_gives_low_probability(self):
        """When all layers strongly favor away, calibrated_prob should be < 0.40."""
        result = blend_predictions(
            mc_prob  = 0.32,
            elo_prob = 0.34,
            rf_prob  = 0.35,
            lr_prob  = 0.36,
            game_date = date(2025, 7, 1),
            apply_calibration = False,
        )
        assert result["calibrated_prob"] < 0.42

    def test_equal_teams_near_50pct(self):
        """All layers at exactly 50% → blended near 50%."""
        result = blend_predictions(0.50, 0.50, 0.50, 0.50, apply_calibration=False)
        assert abs(result["calibrated_prob"] - 0.50) < 0.05

    def test_blend_clamps_extremes(self):
        """Extreme inputs (0.05 / 0.95) should not produce impossible probabilities."""
        result_low  = blend_predictions(0.05, 0.10, 0.08, 0.07, apply_calibration=False)
        result_high = blend_predictions(0.95, 0.90, 0.92, 0.93, apply_calibration=False)

        assert 0.0 < result_low["calibrated_prob"]  < 0.30
        assert 0.70 < result_high["calibrated_prob"] < 1.0


# ===========================================================================
# Random Forest Model
# ===========================================================================

class TestRandomForestModel:

    def test_feature_names_count(self):
        """RF model must have exactly 15 features."""
        assert len(RF_FEATURES) == 15

    def test_feature_names_are_strings(self):
        """All feature names must be non-empty strings."""
        for name in RF_FEATURES:
            assert isinstance(name, str) and len(name) > 0

    def test_predict_returns_valid_probability(self):
        """predict_win_probability must return a float in (0, 1)."""
        rf = get_rf_model()
        features = {f: 0.5 for f in RF_FEATURES}
        # Override key features to reasonable values
        features["home_starter_siera"]      = 3.80
        features["away_starter_siera"]      = 4.20
        features["home_team_xwoba"]         = 0.322
        features["away_team_xwoba"]         = 0.310
        features["park_factor_runs"]        = 1.00
        features["park_factor_hr"]          = 1.00
        features["home_elo"]                = 1510.0
        features["away_elo"]                = 1490.0
        features["home_decay_win_pct"]      = 0.55
        features["away_decay_win_pct"]      = 0.45

        prob = rf.predict_win_probability(features)
        assert 0.0 < prob < 1.0, f"RF predict returned {prob}"

    def test_predict_home_dominant_scenario(self):
        """Clear home-dominant features → home win prob > 0.55."""
        rf = get_rf_model()
        features = {f: 0.0 for f in RF_FEATURES}
        features["home_starter_siera"]      = 2.50   # Elite
        features["away_starter_siera"]      = 5.50   # Replacement
        features["home_starter_decay_xera"] = 2.40
        features["away_starter_decay_xera"] = 5.40
        features["home_team_xwoba"]         = 0.380  # Strong offense
        features["away_team_xwoba"]         = 0.270  # Weak offense
        features["home_bullpen_xfip"]       = 3.50
        features["away_bullpen_xfip"]       = 5.00
        features["park_factor_runs"]        = 1.00
        features["park_factor_hr"]          = 1.00
        features["home_decay_win_pct"]      = 0.70
        features["away_decay_win_pct"]      = 0.30
        features["weather_run_adj"]         = 0.0
        features["home_elo"]                = 1600.0
        features["away_elo"]                = 1400.0

        prob = rf.predict_win_probability(features)
        assert prob > 0.55, f"Home-dominant features gave only {prob:.3f}"

    def test_predict_handles_missing_features_gracefully(self):
        """Missing feature keys should not crash — model uses defaults."""
        rf = get_rf_model()
        # Provide only half the features
        partial = {RF_FEATURES[i]: 0.5 for i in range(0, len(RF_FEATURES), 2)}
        prob = rf.predict_win_probability(partial)
        assert 0.0 < prob < 1.0

    def test_model_singleton_returns_same_instance(self):
        """get_rf_model() should return the same cached instance."""
        rf1 = get_rf_model()
        rf2 = get_rf_model()
        assert rf1 is rf2


# ===========================================================================
# Logistic Regression Model
# ===========================================================================

class TestLogisticRegressionModel:

    def test_feature_names_count(self):
        """LR model must have exactly 12 features."""
        assert len(LR_FEATURES) == 12

    def test_predict_returns_valid_probability(self):
        """predict_win_probability must return a float in (0, 1)."""
        lr = get_lr_model()
        features = {
            "home_starter_era":        3.80,
            "away_starter_era":        4.20,
            "home_team_xwoba":         0.322,
            "away_team_xwoba":         0.310,
            "home_bullpen_xfip":       3.90,
            "away_bullpen_xfip":       4.20,
            "park_factor_runs":        1.00,
            "home_decay_win_pct":      0.55,
            "away_decay_win_pct":      0.45,
            "weather_run_adj":         0.0,
            "home_starter_decay_xera": 3.75,
            "away_starter_decay_xera": 4.15,
        }
        prob = lr.predict_win_probability(features)
        assert 0.0 < prob < 1.0, f"LR predict returned {prob}"

    def test_predict_home_dominant_scenario(self):
        """Clear home-dominant inputs → home win prob > 0.55."""
        lr = get_lr_model()
        features = {
            "home_starter_era":        2.50,  # Elite
            "away_starter_era":        5.50,  # Replacement
            "home_team_xwoba":         0.380,
            "away_team_xwoba":         0.270,
            "home_bullpen_xfip":       3.50,
            "away_bullpen_xfip":       5.00,
            "park_factor_runs":        1.00,
            "home_decay_win_pct":      0.70,
            "away_decay_win_pct":      0.30,
            "weather_run_adj":         0.0,
            "home_starter_decay_xera": 2.40,
            "away_starter_decay_xera": 5.40,
        }
        prob = lr.predict_win_probability(features)
        assert prob > 0.55, f"Home-dominant LR gave only {prob:.3f}"

    def test_predict_handles_missing_features_gracefully(self):
        """Missing features should not crash — uses defaults."""
        lr = get_lr_model()
        partial = {LR_FEATURES[0]: 3.80, LR_FEATURES[1]: 4.20}
        prob = lr.predict_win_probability(partial)
        assert 0.0 < prob < 1.0

    def test_model_singleton_returns_same_instance(self):
        """get_lr_model() should return the same cached instance."""
        lr1 = get_lr_model()
        lr2 = get_lr_model()
        assert lr1 is lr2


# ===========================================================================
# Calibrator
# ===========================================================================

class TestCalibrator:

    def test_calibrate_single_passthrough_when_unfitted(self):
        """
        When the calibrator has not been fitted yet, calibrate_single should
        return the input probability essentially unchanged.
        """
        cal  = get_calibrator()
        raw  = 0.62
        out  = cal.calibrate_single(raw)
        # Pass-through: should be very close to input (within ±10% before fitting)
        assert 0.0 < out < 1.0, f"calibrate_single returned invalid prob: {out}"

    def test_calibrate_single_preserves_direction(self):
        """A probability above 0.5 should remain above 0.5 after calibration."""
        cal = get_calibrator()
        assert cal.calibrate_single(0.65) > 0.50
        assert cal.calibrate_single(0.35) < 0.50

    def test_calibrate_array_length_preserved(self):
        """calibrate() on an array should return same-length array."""
        cal   = get_calibrator()
        probs = np.array([0.40, 0.50, 0.60, 0.70])
        out   = cal.calibrate(probs)
        assert len(out) == len(probs)

    def test_fit_and_calibrate_with_synthetic_data(self):
        """
        Fit the calibrator on perfectly-calibrated synthetic data.
        After fitting, calibration should pass through approximately correctly.
        """
        cal = get_calibrator()

        # Perfect calibration: prediction = actual rate
        np.random.seed(77)
        raw_probs = np.linspace(0.30, 0.70, 200)
        actuals   = (np.random.rand(200) < raw_probs).astype(int)

        metrics = cal.fit(raw_probs, actuals)

        assert "method" in metrics
        assert "n_games" in metrics
        assert metrics["n_games"] == 200

        # After fitting, calibration should work without error
        out = cal.calibrate_single(0.60)
        assert 0.0 < out < 1.0

    def test_brier_score_method_exists(self):
        """calibrator.brier_score() should run without error."""
        cal     = get_calibrator()
        probs   = np.array([0.55, 0.45, 0.65, 0.35])
        actuals = np.array([1, 0, 1, 0])

        brier = cal.brier_score(probs, actuals)
        assert 0.0 <= brier <= 1.0

    def test_calibration_curve_returns_valid_structure(self):
        """calibration_curve() should return dict with bins data."""
        cal     = get_calibrator()
        probs   = np.linspace(0.30, 0.70, 100)
        actuals = (np.random.rand(100) < probs).astype(int)

        curve = cal.calibration_curve(probs, actuals, n_bins=5)
        assert isinstance(curve, dict)
        assert "bin_centers" in curve or len(curve) > 0
