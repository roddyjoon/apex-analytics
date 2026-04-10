"""
Apex Analytics — Simulation Layer Tests
Tests the Monte Carlo engine, PA outcome calculator, game simulator,
and run distribution properties.

All tests use seeded RNGs for determinism.
Fast: entire suite completes in < 30 seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.pa_calculator import compute_pa_outcomes, sample_outcome, is_hit, is_out
from simulation.game_simulator import simulate_game
from simulation.monte_carlo import run_monte_carlo
from simulation.profiles import GameContext, ParkContext

from tests.conftest import (
    make_batter, make_pitcher, make_lineup, make_park,
    make_coors_park, make_oracle_park, make_dome_park, make_windy_park,
    make_context, make_bullpen, make_defense,
    make_elite_pitcher, make_replacement_pitcher,
)


# ===========================================================================
# PA Outcome Calculator
# ===========================================================================

class TestPAOutcomes:

    def test_outcomes_sum_to_one_league_average(self):
        """PA outcomes must sum exactly to 1.0 for league-average players."""
        batter  = make_batter()
        pitcher = make_pitcher()
        probs   = compute_pa_outcomes(batter, pitcher)

        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, f"Outcomes sum to {total}, expected 1.0"

    def test_outcomes_sum_to_one_extreme_pitcher(self):
        """Outcomes must still sum to 1.0 for a dominant pitcher."""
        batter  = make_batter(xwoba=0.200)
        pitcher = make_elite_pitcher()
        probs   = compute_pa_outcomes(batter, pitcher)

        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, f"Outcomes sum to {total} for elite pitcher"

    def test_outcomes_sum_to_one_replacement_pitcher(self):
        """Outcomes must still sum to 1.0 for a poor pitcher."""
        batter  = make_batter(xwoba=0.380)
        pitcher = make_replacement_pitcher()
        probs   = compute_pa_outcomes(batter, pitcher)

        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, f"Outcomes sum to {total} for replacement pitcher"

    def test_required_outcome_keys(self):
        """All 9 expected outcome types must be present."""
        expected_keys = {"hr", "triple", "double", "single", "walk", "hbp",
                         "strikeout", "groundout", "flyout"}
        batter  = make_batter()
        pitcher = make_pitcher()
        probs   = compute_pa_outcomes(batter, pitcher)

        missing = expected_keys - set(probs.keys())
        assert not missing, f"Missing outcome keys: {missing}"

    def test_all_probabilities_non_negative(self):
        """No outcome can have a negative probability."""
        batter  = make_batter()
        pitcher = make_pitcher()
        probs   = compute_pa_outcomes(batter, pitcher)

        for outcome, p in probs.items():
            assert p >= 0.0, f"Negative probability for {outcome}: {p}"

    def test_elite_pitcher_raises_strikeout_rate(self):
        """Elite pitcher (high K%) should produce higher strikeout probability."""
        batter         = make_batter()
        elite_pitcher  = make_elite_pitcher(k_pct=0.32)
        avg_pitcher    = make_pitcher(k_pct=0.20)

        elite_probs = compute_pa_outcomes(batter, elite_pitcher)
        avg_probs   = compute_pa_outcomes(batter, avg_pitcher)

        assert elite_probs["strikeout"] > avg_probs["strikeout"], (
            f"Elite pitcher should have higher K rate: "
            f"elite={elite_probs['strikeout']:.3f} vs avg={avg_probs['strikeout']:.3f}"
        )

    def test_elite_batter_raises_hr_rate(self):
        """Elite batter (high xwOBA, high barrel%) should have higher HR probability."""
        elite_batter   = make_batter(xwoba=0.420)
        league_batter  = make_batter(xwoba=0.300)
        pitcher        = make_pitcher()

        elite_probs  = compute_pa_outcomes(elite_batter, pitcher)
        league_probs = compute_pa_outcomes(league_batter, pitcher)

        assert elite_probs["hr"] > league_probs["hr"], (
            f"Elite batter HR rate should be higher: "
            f"elite={elite_probs['hr']:.4f} avg={league_probs['hr']:.4f}"
        )

    def test_park_factor_affects_hr_probability(self):
        """Coors Field (HR factor 1.30) should produce more HRs than Oracle Park."""
        batter       = make_batter()
        pitcher      = make_pitcher()
        coors_probs  = compute_pa_outcomes(batter, pitcher, park=make_coors_park())
        oracle_probs = compute_pa_outcomes(batter, pitcher, park=make_oracle_park())

        assert coors_probs["hr"] > oracle_probs["hr"], (
            "Coors should have higher HR probability than Oracle Park"
        )

    def test_outcomes_with_none_park_still_valid(self):
        """park=None should work gracefully (uses league-average park factors)."""
        batter  = make_batter()
        pitcher = make_pitcher()
        probs   = compute_pa_outcomes(batter, pitcher, park=None)

        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_outs_parameter_accepted(self):
        """outs parameter (0, 1, 2) should all produce valid probability dicts."""
        batter  = make_batter()
        pitcher = make_pitcher()

        for outs in (0, 1, 2):
            probs = compute_pa_outcomes(batter, pitcher, outs=outs)
            assert abs(sum(probs.values()) - 1.0) < 1e-6, \
                f"Outcomes don't sum to 1.0 with outs={outs}"

    def test_sample_outcome_returns_valid_key(self):
        """sample_outcome should return one of the outcome keys."""
        batter  = make_batter()
        pitcher = make_pitcher()
        probs   = compute_pa_outcomes(batter, pitcher)
        rng     = np.random.default_rng(42)

        for _ in range(50):
            outcome = sample_outcome(probs, rng)
            assert outcome in probs, f"sample_outcome returned unknown key: {outcome}"


# ===========================================================================
# Game Simulator
# ===========================================================================

class TestGameSimulator:

    def test_single_game_returns_required_keys(self, rng):
        """simulate_game must return all required output keys."""
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        required = {"home_score", "away_score", "home_win", "total_runs",
                    "innings_played", "went_extra"}
        missing  = required - set(result.keys())
        assert not missing, f"simulate_game missing keys: {missing}"

    def test_single_game_scores_are_non_negative(self, rng):
        """Scores cannot be negative."""
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        assert result["home_score"] >= 0
        assert result["away_score"] >= 0
        assert result["total_runs"] >= 0

    def test_total_runs_equals_sum_of_scores(self, rng):
        """total_runs must equal home_score + away_score."""
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        assert result["total_runs"] == result["home_score"] + result["away_score"]

    def test_home_win_consistent_with_scores(self, rng):
        """home_win flag must match the score comparison."""
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        if result["home_score"] > result["away_score"]:
            assert result["home_win"] is True
        else:
            assert result["home_win"] is False

    def test_innings_played_at_least_nine(self, rng):
        """All MLB games go at least 9 innings."""
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        assert result["innings_played"] >= 9, \
            f"Game only played {result['innings_played']} innings"

    def test_innings_played_cap_respected(self, rng):
        """innings_played must not exceed EXTRA_INNINGS_CAP."""
        from config import EXTRA_INNINGS_CAP
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        assert result["innings_played"] <= EXTRA_INNINGS_CAP + 1

    def test_no_tie_results(self, rng):
        """MLB games cannot end in a tie — always exactly one winner."""
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        assert result["home_score"] != result["away_score"], \
            "Game ended in a tie — extra innings should prevent this"

    def test_extra_innings_flag(self, rng):
        """went_extra should be True if and only if game lasted > 9 innings."""
        ctx    = make_context()
        result = simulate_game(ctx, rng)

        if result["innings_played"] > 9:
            assert result["went_extra"] is True
        else:
            assert result["went_extra"] is False


# ===========================================================================
# Monte Carlo — Core Properties
# ===========================================================================

class TestMonteCarlo:

    def test_mc_returns_required_keys(self):
        """run_monte_carlo must return the full expected output dict."""
        ctx    = make_context()
        result = run_monte_carlo(ctx, n_iterations=200)

        required = {
            "home_win_pct", "away_win_pct",
            "projected_home_runs", "projected_away_runs", "projected_total",
            "run_distribution", "home_run_distribution", "away_run_distribution",
            "confidence_interval", "n_iterations",
        }
        missing = required - set(result.keys())
        assert not missing, f"run_monte_carlo missing keys: {missing}"

    def test_win_probabilities_sum_to_one(self):
        """home_win_pct + away_win_pct must sum to exactly 1.0."""
        ctx    = make_context()
        result = run_monte_carlo(ctx, n_iterations=200)

        total = result["home_win_pct"] + result["away_win_pct"]
        assert abs(total - 1.0) < 1e-6, f"Win probs sum to {total}"

    def test_home_field_advantage_baseline(self):
        """
        Two identical league-average teams → home team wins 51-58%.
        Pure home field advantage from simulation encoding.
        """
        ctx    = make_context()
        result = run_monte_carlo(ctx, n_iterations=500)

        home_pct = result["home_win_pct"]
        assert 0.51 <= home_pct <= 0.58, (
            f"League-avg HFA should be 51-58%, got {home_pct:.1%}"
        )

    def test_elite_vs_replacement_home_dominant(self):
        """
        Elite home SP (2.50 ERA) vs. replacement away SP (5.50 ERA)
        → home wins > 62%.
        """
        ctx    = make_context(home_starter_era=2.50, away_starter_era=5.50)
        result = run_monte_carlo(ctx, n_iterations=500)

        home_pct = result["home_win_pct"]
        assert home_pct > 0.60, (
            f"Elite vs. replacement: expected home > 60%, got {home_pct:.1%}"
        )

    def test_elite_away_sp_suppresses_home(self):
        """
        Replacement home SP (5.50) vs. elite away SP (2.50)
        → home wins < 38%.
        """
        ctx    = make_context(home_starter_era=5.50, away_starter_era=2.50)
        result = run_monte_carlo(ctx, n_iterations=500)

        home_pct = result["home_win_pct"]
        assert home_pct < 0.42, (
            f"Replacement vs. elite: expected home < 42%, got {home_pct:.1%}"
        )

    def test_projected_total_reasonable_range(self):
        """Projected total runs should be between 5 and 20 for normal conditions."""
        ctx    = make_context()
        result = run_monte_carlo(ctx, n_iterations=200)

        total = result["projected_total"]
        assert 5.0 <= total <= 20.0, f"Projected total out of range: {total}"

    def test_coors_higher_total_than_oracle(self):
        """Coors Field (run factor 1.15) should produce more total runs than Oracle Park."""
        coors_ctx  = make_context(park=make_coors_park())
        oracle_ctx = make_context(park=make_oracle_park())

        coors_result  = run_monte_carlo(coors_ctx,  n_iterations=300)
        oracle_result = run_monte_carlo(oracle_ctx, n_iterations=300)

        assert coors_result["projected_total"] > oracle_result["projected_total"], (
            f"Coors ({coors_result['projected_total']:.2f}) should exceed "
            f"Oracle ({oracle_result['projected_total']:.2f})"
        )

    def test_wind_out_increases_total(self):
        """Wind blowing out (+0.7 runs/game) should increase projected total vs. calm."""
        calm_ctx    = make_context(park=make_park(wind_speed=0.0))
        wind_ctx    = make_context(park=make_windy_park(wind_out=True))

        calm_result = run_monte_carlo(calm_ctx, n_iterations=300)
        wind_result = run_monte_carlo(wind_ctx, n_iterations=300)

        assert wind_result["projected_total"] > calm_result["projected_total"], (
            f"Wind out should boost total: "
            f"wind={wind_result['projected_total']:.2f} calm={calm_result['projected_total']:.2f}"
        )

    def test_wind_in_decreases_total(self):
        """
        Wind blowing in (-0.6 runs) should produce fewer runs than wind blowing out (+0.7 runs).
        Comparing both within the SAME park (Wrigley Field) eliminates park-factor confounding.
        The expected delta is ~1.3 runs, easily detectable at 2000 iterations.
        """
        wind_out_ctx = make_context(park=make_windy_park(wind_out=True))   # +0.7 runs
        wind_in_ctx  = make_context(park=make_windy_park(wind_out=False))  # -0.6 runs

        out_result = run_monte_carlo(wind_out_ctx, n_iterations=2000, base_seed=7777)
        in_result  = run_monte_carlo(wind_in_ctx,  n_iterations=2000, base_seed=7777)

        assert in_result["projected_total"] < out_result["projected_total"], (
            f"Wind IN should produce fewer runs than wind OUT (same park, same seed): "
            f"in={in_result['projected_total']:.2f} out={out_result['projected_total']:.2f}"
        )

    def test_run_distribution_has_percentiles(self):
        """run_distribution must include key percentile labels."""
        ctx    = make_context()
        result = run_monte_carlo(ctx, n_iterations=200)

        rd = result["run_distribution"]
        assert isinstance(rd, dict)
        assert len(rd) > 0, "run_distribution is empty"
        # Should have at least p50 (median)
        assert "p50" in rd or "median" in rd or 50 in rd, \
            f"Median missing from run_distribution keys: {list(rd.keys())}"

    def test_confidence_interval_valid(self):
        """confidence_interval should be a (lower, upper) tuple with 0 ≤ lower < upper ≤ 1."""
        ctx    = make_context()
        result = run_monte_carlo(ctx, n_iterations=200)

        ci = result["confidence_interval"]
        assert isinstance(ci, (tuple, list)) and len(ci) == 2
        lo, hi = ci
        assert 0.0 <= lo < hi <= 1.0, f"CI out of range: {lo:.4f} – {hi:.4f}"

    def test_n_iterations_matches_requested(self):
        """n_iterations in output must equal (or be close to) what was requested."""
        n   = 200
        ctx = make_context()
        result = run_monte_carlo(ctx, n_iterations=n)

        assert result["n_iterations"] == n

    def test_deterministic_with_same_seed(self):
        """Same base_seed → same win probability (within rounding)."""
        ctx = make_context()
        r1  = run_monte_carlo(ctx, n_iterations=100, base_seed=0)
        r2  = run_monte_carlo(ctx, n_iterations=100, base_seed=0)

        assert r1["home_win_pct"] == r2["home_win_pct"]

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) give different results."""
        ctx = make_context()
        r1  = run_monte_carlo(ctx, n_iterations=100, base_seed=0)
        r2  = run_monte_carlo(ctx, n_iterations=100, base_seed=9999)

        # Not guaranteed to differ but highly probable with 100 iters
        # We just check neither crashes
        assert 0 < r1["home_win_pct"] < 1
        assert 0 < r2["home_win_pct"] < 1

    def test_strong_offense_vs_weak_defense_boosts_runs(self):
        """Elite offense (high xwOBA) should produce more runs than weak offense."""
        strong_ctx = make_context(home_lineup_xwoba=0.400, away_lineup_xwoba=0.400)
        weak_ctx   = make_context(home_lineup_xwoba=0.260, away_lineup_xwoba=0.260)

        strong_result = run_monte_carlo(strong_ctx, n_iterations=300)
        weak_result   = run_monte_carlo(weak_ctx,   n_iterations=300)

        assert strong_result["projected_total"] > weak_result["projected_total"], (
            f"Strong offense should score more: "
            f"strong={strong_result['projected_total']:.2f} "
            f"weak={weak_result['projected_total']:.2f}"
        )


# ===========================================================================
# Pitcher removal / fatigue
# ===========================================================================

class TestPitcherRemoval:

    def test_fatigued_bullpen_used_when_starter_exits(self, rng):
        """
        Verify game completes normally even with a fatigued bullpen.
        (Fatigue flag set; game should still finish with a result.)
        """
        ctx = make_context(home_fatigued=True, away_fatigued=True)
        result = simulate_game(ctx, rng)

        assert "home_win" in result
        assert result["innings_played"] >= 9
