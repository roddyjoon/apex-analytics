"""
Shared pytest fixtures for Apex Analytics test suite.
Provides factory functions for all simulation dataclasses so individual
test files don't duplicate boilerplate.
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.profiles import (
    BatterProfile,
    BullpenProfile,
    DefenseProfile,
    GameContext,
    ParkContext,
    PitcherProfile,
)


# ---------------------------------------------------------------------------
# Pitcher factories
# ---------------------------------------------------------------------------

def make_pitcher(
    era: float = 4.20,
    k_pct: float = 0.22,
    bb_pct: float = 0.08,
    throws: str = "R",
    is_starter: bool = True,
    name: str = "Test Pitcher",
    player_id: int = 999001,
    is_home: bool = True,
) -> PitcherProfile:
    """Return a valid PitcherProfile with sensible defaults."""
    return PitcherProfile(
        player_id       = player_id,
        player_name     = name,
        throws          = throws,
        is_starter      = is_starter,
        is_opener       = False,
        true_talent_era = era,
        xera            = era - 0.10,
        fip             = era + 0.05,
        siera           = era - 0.05,
        k_pct           = k_pct,
        bb_pct          = bb_pct,
        gb_pct          = 0.44,
        fb_pct          = 0.36,
        ld_pct          = 0.20,
        swstr_pct       = 0.115,
        csw_pct         = 0.280,
        barrel_pct_allowed = 0.080,
        xba_allowed     = 0.248,
        home_era_adj    = 1.00,
        away_era_adj    = 1.00,
        exp_decay_xera  = era - 0.10,
        exp_decay_k_pct = k_pct,
        exp_decay_bb_pct = bb_pct,
        arsenal         = {"FF": 0.55, "SL": 0.25, "CH": 0.15, "CU": 0.05},
        fatigue_index   = 0.0,
        days_rest       = 4,
        pitch_count_est = 0,
        stuff_plus_proxy = 100.0,
        bf              = 300,
        confidence      = "HIGH",
        is_on_il        = False,
        is_home         = is_home,
    )


def make_elite_pitcher(**kwargs) -> PitcherProfile:
    """Ace-caliber SP (sub-3.00 ERA)."""
    defaults = dict(era=2.50, k_pct=0.32, bb_pct=0.06, name="Elite SP")
    defaults.update(kwargs)
    return make_pitcher(**defaults)


def make_replacement_pitcher(**kwargs) -> PitcherProfile:
    """Replacement-level SP (5.50+ ERA)."""
    defaults = dict(era=5.50, k_pct=0.16, bb_pct=0.12, name="Replacement SP")
    defaults.update(kwargs)
    return make_pitcher(**defaults)


# ---------------------------------------------------------------------------
# Batter factories
# ---------------------------------------------------------------------------

def make_batter(
    xwoba: float = 0.320,
    pa: int = 300,
    order: int = 1,
    bats: str = "R",
    name: str = "Test Batter",
    player_id: int = 888001,
    position: str = "CF",
    is_confirmed: bool = True,
) -> BatterProfile:
    """Return a valid BatterProfile with sensible defaults."""
    return BatterProfile(
        player_id       = player_id,
        player_name     = name,
        batting_order   = order,
        position        = position,
        bats            = bats,
        xwoba           = xwoba,
        xba             = 0.248,
        barrel_pct      = 0.080,
        hard_hit_pct    = 0.380,
        swstr_pct       = 0.115,
        k_pct           = 0.228,
        bb_pct          = 0.083,
        hr_rate         = 0.037,
        obp             = 0.315,
        slg             = 0.413,
        sprint_speed    = 27.0,
        xwoba_vs_lhp    = xwoba,
        xwoba_vs_rhp    = xwoba,
        pa_vs_lhp       = pa // 2,
        pa_vs_rhp       = pa // 2,
        exp_decay_xwoba = xwoba,
        pitch_type_xwoba = {},
        matchup_history = {},
        pa              = pa,
        confidence      = "HIGH" if pa >= 50 else "LOW",
        is_confirmed    = is_confirmed,
        is_projected    = not is_confirmed,
        pa_season       = pa,
    )


def make_lineup(
    n: int = 9,
    xwoba: float = 0.320,
    pa: int = 300,
    confirmed: bool = True,
) -> list:
    """Return an n-batter lineup with uniform xwOBA."""
    positions = ["CF", "SS", "RF", "1B", "DH", "3B", "LF", "C", "2B"]
    return [
        make_batter(
            xwoba        = xwoba,
            pa           = pa,
            order        = i + 1,
            name         = f"Batter {i + 1}",
            player_id    = 800000 + i,
            position     = positions[i % len(positions)],
            is_confirmed = confirmed,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Bullpen factory
# ---------------------------------------------------------------------------

def make_bullpen(
    xfip: float = 4.10,
    ip_available: float = 4.0,
    fatigued: bool = False,
    team_id: int = 147,
    team_abbr: str = "NYY",
) -> BullpenProfile:
    return BullpenProfile(
        team_id          = team_id,
        team_abbr        = team_abbr,
        xfip             = xfip,
        high_lev_xfip   = xfip - 0.20,
        era              = xfip + 0.15,
        ip_available     = ip_available,
        fatigue_flag     = fatigued,
        prior_game_innings = 11 if fatigued else 9,
        pct_righties     = 0.60,
        pct_lefties      = 0.40,
    )


# ---------------------------------------------------------------------------
# Park factory
# ---------------------------------------------------------------------------

def make_park(
    run_factor: float = 1.00,
    hr_factor: float = 1.00,
    temp_f: float = 72.0,
    wind_speed: float = 5.0,
    is_dome: bool = False,
    team_abbr: str = "NYY",
    net_run_adj: float = 0.0,
) -> ParkContext:
    return ParkContext(
        venue_id          = 3313,
        venue_name        = "Test Stadium",
        team_abbr         = team_abbr,
        run_factor        = run_factor,
        hr_factor         = hr_factor,
        temp_f            = temp_f,
        wind_speed_mph    = wind_speed,
        wind_classification = "CALM",
        wind_run_adj      = 0.0,
        temp_run_adj      = 0.0,
        net_run_adj       = net_run_adj,
        weather_note      = "",
        is_dome           = is_dome,
        elevation_ft      = 0,
    )


def make_coors_park() -> ParkContext:
    return make_park(run_factor=1.15, hr_factor=1.30, temp_f=70, team_abbr="COL")


def make_oracle_park() -> ParkContext:
    return make_park(run_factor=0.91, hr_factor=0.83, temp_f=60, team_abbr="SF")


def make_dome_park() -> ParkContext:
    return make_park(is_dome=True, team_abbr="TB")


def make_windy_park(wind_out: bool = True) -> ParkContext:
    """Simulate wind blowing out (+0.7 runs) or in (-0.6 runs)."""
    adj = +0.7 if wind_out else -0.6
    return ParkContext(
        venue_id          = 17,
        venue_name        = "Wrigley Field",
        team_abbr         = "CHC",
        run_factor        = 1.04,
        hr_factor         = 1.065,
        temp_f            = 72.0,
        wind_speed_mph    = 16.0,
        # Use lowercase to match weather.py _classify_wind output and pa_calculator checks
        wind_classification = "out" if wind_out else "in",
        wind_run_adj      = adj,
        temp_run_adj      = 0.0,
        net_run_adj       = adj,
        weather_note      = "Wind blowing out 16mph" if wind_out else "Wind blowing in 16mph",
        is_dome           = False,
        elevation_ft      = 595,
    )


# ---------------------------------------------------------------------------
# Defense factory
# ---------------------------------------------------------------------------

def make_defense() -> DefenseProfile:
    return DefenseProfile(
        team_id   = 147,
        team_abbr = "NYY",
        drs_runs  = 0.0,
        uzr       = 0.0,
        babip_adj = 0.0,
    )


# ---------------------------------------------------------------------------
# Full GameContext factory
# ---------------------------------------------------------------------------

def make_context(
    home_starter_era:  float = 4.20,
    away_starter_era:  float = 4.20,
    home_lineup_xwoba: float = 0.320,
    away_lineup_xwoba: float = 0.320,
    home_bull_xfip:    float = 4.10,
    away_bull_xfip:    float = 4.10,
    park:              ParkContext = None,
    home_elo:          float = 1500.0,
    away_elo:          float = 1500.0,
    home_fatigued:     bool = False,
    away_fatigued:     bool = False,
    n_batters:         int = 9,
    game_pk:           int = 12345,
) -> GameContext:
    """Return a fully populated, minimal-valid GameContext for testing."""
    if park is None:
        park = make_park()

    return GameContext(
        game_pk          = game_pk,
        game_date        = date(2025, 6, 15),
        game_time_utc    = datetime(2025, 6, 15, 23, 10, tzinfo=timezone.utc),
        home_team_id     = 147,
        home_team_abbr   = "NYY",
        away_team_id     = 119,
        away_team_abbr   = "LAD",
        home_starter     = make_pitcher(era=home_starter_era, is_home=True),
        away_starter     = make_pitcher(era=away_starter_era, is_home=False),
        home_lineup      = make_lineup(n=n_batters, xwoba=home_lineup_xwoba),
        away_lineup      = make_lineup(n=n_batters, xwoba=away_lineup_xwoba),
        home_bullpen     = make_bullpen(xfip=home_bull_xfip, fatigued=home_fatigued,
                                        team_id=147, team_abbr="NYY"),
        away_bullpen     = make_bullpen(xfip=away_bull_xfip, fatigued=away_fatigued,
                                        team_id=119, team_abbr="LAD"),
        home_defense     = make_defense(),
        away_defense     = make_defense(),
        park             = park,
        home_elo         = home_elo,
        away_elo         = away_elo,
        home_decay_win_pct = 0.500,
        away_decay_win_pct = 0.500,
        lineup_source_home = "confirmed",
        lineup_source_away = "confirmed",
        report_type      = "morning",
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for deterministic simulation tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def league_avg_context() -> GameContext:
    """Two league-average teams in a neutral park."""
    return make_context()


@pytest.fixture
def elite_vs_replacement_context() -> GameContext:
    """Elite home SP (2.50 ERA) vs. replacement-level away SP (5.50 ERA)."""
    return make_context(home_starter_era=2.50, away_starter_era=5.50)


@pytest.fixture
def coors_context() -> GameContext:
    return make_context(park=make_coors_park())


@pytest.fixture
def oracle_context() -> GameContext:
    return make_context(park=make_oracle_park())


@pytest.fixture
def dome_context() -> GameContext:
    return make_context(park=make_dome_park())


@pytest.fixture
def wind_out_context() -> GameContext:
    return make_context(park=make_windy_park(wind_out=True))


@pytest.fixture
def wind_in_context() -> GameContext:
    return make_context(park=make_windy_park(wind_out=False))
