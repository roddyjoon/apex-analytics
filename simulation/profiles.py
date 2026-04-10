"""
Apex Analytics — Simulation Profile Dataclasses
All profile objects used throughout the simulation engine.
These are the contracts between the data layer and the Monte Carlo simulator.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BatterProfile:
    """
    Complete batter profile for one plate appearance in the simulation.
    All stats are Bayesian-blended (prior season + current season).
    """
    # Identity
    player_id:    int
    player_name:  str
    batting_order: int                     # 1-9
    position:     str                      # CF, SS, 1B, DH, etc.
    bats:         str = "R"               # L / R / S

    # Core Statcast metrics (season-blended)
    xwoba:        float = 0.320           # Primary contact quality driver
    xba:          float = 0.250           # BABIP-adjusted hit probability
    barrel_pct:   float = 0.080           # HR / XBH elevation
    hard_hit_pct: float = 0.380           # Contact quality multiplier
    swstr_pct:    float = 0.115           # Whiff rate (strikeout susceptibility)
    k_pct:        float = 0.228           # Strikeout rate
    bb_pct:       float = 0.083           # Walk rate
    hr_rate:      float = 0.037           # HR per PA
    obp:          float = 0.315
    slg:          float = 0.413
    sprint_speed: float = 27.0            # ft/s — base running

    # Platoon splits
    xwoba_vs_lhp: float = 0.320
    xwoba_vs_rhp: float = 0.320
    pa_vs_lhp:    int = 0
    pa_vs_rhp:    int = 0

    # Recency-weighted (exponential decay, 21-day window)
    exp_decay_xwoba:  Optional[float] = None   # None = insufficient recent data

    # Pitch-type vulnerability (xwOBA vs specific pitch types)
    # Keys: "FF", "SL", "CH", "CU", "SI", "KC", "FC"
    pitch_type_xwoba: dict = field(default_factory=dict)

    # Batter-vs-pitcher matchup history
    # Keys: pitcher_id (int) → {"pa", "xwoba", "weight"}
    matchup_history: dict = field(default_factory=dict)

    # Sample size / confidence
    pa:           int = 0                 # Season PA count
    confidence:   str = "LOW"            # HIGH / MEDIUM / LOW
    is_confirmed: bool = False            # In confirmed lineup vs projected
    is_projected: bool = True             # Lineup source flag

    # PA count before blending (for Bayesian weight)
    pa_season:    int = 0


@dataclass
class PitcherProfile:
    """
    Complete pitcher profile for the simulation engine.
    Contains true-talent ERA blend, SIERA inputs, pitch arsenal,
    fatigue state, and location adjustments.
    """
    # Identity
    player_id:    int
    player_name:  str
    throws:       str = "R"              # L / R
    is_starter:   bool = True
    is_opener:    bool = False           # Bullpen game / opener detection

    # True-talent ERA blend: SIERA×0.45 + xERA×0.35 + FIP×0.20
    true_talent_era:  float = 4.20
    xera:             float = 4.20
    fip:              float = 4.20
    siera:            float = 4.20

    # Raw SIERA inputs (for re-computation if needed)
    k_pct:    float = 0.228
    bb_pct:   float = 0.083
    gb_pct:   float = 0.430
    fb_pct:   float = 0.350
    ld_pct:   float = 0.220

    # Stuff quality
    swstr_pct:  float = 0.115           # Swinging strike rate
    csw_pct:    float = 0.280           # Called strike + whiff %

    # Contact quality allowed
    barrel_pct_allowed: float = 0.080
    xba_allowed:        float = 0.250

    # Home / away ERA adjustment multipliers (1.0 = neutral)
    home_era_adj: float = 1.0
    away_era_adj: float = 1.0

    # Recency-weighted (last 5 starts, exponential decay)
    exp_decay_xera:    Optional[float] = None
    exp_decay_k_pct:   Optional[float] = None
    exp_decay_bb_pct:  Optional[float] = None

    # Pitch arsenal: {pitch_type → {usage_pct, xwoba_conceded, swstr_pct, avg_velocity}}
    arsenal: dict = field(default_factory=dict)

    # Fatigue state (computed by simulation engine per inning)
    fatigue_index:    float = 0.0       # Increases per inning; see FATIGUE_PER_INNING
    days_rest:        int = 4
    pitch_count_est:  int = 0           # Running estimate during simulation

    # Stuff+ proxy (swstr_pct relative to league avg; used for removal probability)
    stuff_plus_proxy: float = 100.0

    # Sample size / confidence
    bf:           int = 0               # Batters faced this season
    confidence:   str = "LOW"
    is_on_il:     bool = False

    # Context
    is_home:      bool = False          # Set by game context


@dataclass
class BullpenProfile:
    """
    Team bullpen aggregate profile.
    Used from the moment the starter is removed.
    """
    team_id:      int
    team_abbr:    str

    # Aggregate quality
    xfip:               float = 4.20    # True-talent bullpen xFIP
    high_lev_xfip:      float = 4.00    # High-leverage relievers only
    era:                float = 4.20

    # Fatigue state
    ip_available:       float = 4.0     # Estimated available innings
    fatigue_flag:       bool = False    # True if prior game went extra innings
    prior_game_innings: int = 9         # Innings in previous game

    # Handedness mix
    pct_righties:  float = 0.60
    pct_lefties:   float = 0.40


@dataclass
class DefenseProfile:
    """
    Team defense profile — affects BABIP / out rate in simulation.
    Currently minimal; placeholder for DRS/UZR integration.
    """
    team_id:    int
    team_abbr:  str
    drs_runs:   float = 0.0            # Defensive Runs Saved vs average
    uzr:        float = 0.0            # UZR/150
    # Rough translation: every 10 DRS ≈ ±0.012 BABIP adjustment
    babip_adj:  float = 0.0


@dataclass
class ParkContext:
    """
    Park factor and weather context for a game.
    Applied globally to all PA outcomes in the simulation.
    """
    venue_id:     int
    venue_name:   str
    team_abbr:    str

    # Park factors
    run_factor:   float = 1.0          # Runs environment (1.0 = neutral)
    hr_factor:    float = 1.0          # HR environment

    # Weather adjustments (computed by weather.py)
    temp_f:                float = 72.0
    wind_speed_mph:        float = 0.0
    wind_classification:   str = "calm"  # "out" / "in" / "cross" / "calm"
    wind_run_adj:          float = 0.0
    temp_run_adj:          float = 0.0
    net_run_adj:           float = 0.0
    weather_note:          str = ""
    is_dome:               bool = False

    # Elevation (Coors Field altitude effect built into park factors,
    # but tracked here for report display)
    elevation_ft: int = 0


@dataclass
class GameContext:
    """
    Full game context object — passed into the Monte Carlo engine.
    Assembles all sub-profiles for one game.
    """
    # Game identity
    game_pk:    int
    game_date:  str                    # ISO format YYYY-MM-DD
    game_time_utc: Optional[str] = None

    # Teams
    home_team_id:   int = 0
    home_team_abbr: str = ""
    away_team_id:   int = 0
    away_team_abbr: str = ""

    # Starting pitchers
    home_starter:   Optional[PitcherProfile] = None
    away_starter:   Optional[PitcherProfile] = None

    # Lineups (batting order 1-9)
    home_lineup:    list[BatterProfile] = field(default_factory=list)
    away_lineup:    list[BatterProfile] = field(default_factory=list)

    # Bullpens
    home_bullpen:   Optional[BullpenProfile] = None
    away_bullpen:   Optional[BullpenProfile] = None

    # Defense
    home_defense:   Optional[DefenseProfile] = None
    away_defense:   Optional[DefenseProfile] = None

    # Park + weather
    park:           Optional[ParkContext] = None

    # Ensemble inputs (populated by pipeline before blending)
    home_elo:             float = 1500.0   # Home team Elo rating
    away_elo:             float = 1500.0   # Away team Elo rating
    home_decay_win_pct:   Optional[float] = None  # Exp-decay win% last 15 games
    away_decay_win_pct:   Optional[float] = None

    # Report metadata
    lineup_source_home: str = "projected"   # "confirmed" / "historical" / "roster"
    lineup_source_away: str = "projected"
    report_type:    str = "morning"         # "morning" / "pregame"
