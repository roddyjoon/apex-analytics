"""
Apex Analytics — Central Configuration
All constants, thresholds, and tunable parameters live here.
Change a number here; it propagates everywhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR              = Path(__file__).parent
REPORT_OUTPUT_DIR     = Path(os.getenv("REPORT_OUTPUT_DIR", "./reports"))
CACHE_DIR             = Path(os.getenv("CACHE_DIR", "./cache"))
MODELS_DIR            = BASE_DIR / "models"
DATA_DIR              = BASE_DIR / "data"
PRIOR_SEASON_DIR      = DATA_DIR / "prior_season"
HISTORICAL_DIR        = DATA_DIR / "historical"
STADIUMS_FILE         = DATA_DIR / "stadiums.json"

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_URL          = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/apex.db")

# =============================================================================
# SCHEDULE
# =============================================================================
TIMEZONE              = os.getenv("TIMEZONE", "America/Los_Angeles")
REPORT_MORNING_TIME   = os.getenv("REPORT_MORNING_TIME", "08:00")   # PT — projected lineups
REPORT_PREGAME_TIME   = os.getenv("REPORT_PREGAME_TIME", "13:00")   # PT — confirmed lineups

# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================
MONTE_CARLO_ITERATIONS     = 7_000
MAX_STARTER_INNINGS        = 7          # Hard ceiling; probabilistic removal below
PITCHER_PITCH_COUNT_LIMIT  = 100        # Hard removal at this estimated pitch count
PITCHES_PER_INNING_BASE    = 15         # Base pitch estimate per inning
PITCHES_PER_INNING_SLOPE   = 0.5        # Pitches increase as fatigue builds (per inning)
FATIGUE_PER_INNING         = 0.015      # Fatigue index added each inning
MAX_FATIGUE_PENALTY        = 0.15       # Max 15% ERA degradation from fatigue
HOME_FIELD_ADVANTAGE       = 0.035      # ~3.5% win probability boost for home team

# Probabilistic starter removal by inning (manager model)
STARTER_REMOVAL_PROBS = {
    4: 0.05,
    5: 0.20,
    6: 0.55,
    7: 0.90,
}

# Extra innings ghost runner model
EXTRA_INNINGS_SCORING_RATE = 0.60       # Poisson lambda for runs/half-inning with ghost runner
EXTRA_INNINGS_CAP          = 13         # Max innings simulated before forcing resolution

# =============================================================================
# ELO SYSTEM
# =============================================================================
ELO_STARTING               = 1500
ELO_HOME_FIELD_BONUS       = 48         # Points added to home team Elo for win probability
ELO_K_FACTOR               = 20         # Base rating volatility
ELO_SEASON_REGRESSION      = 0.33       # Regress 33% toward 1500 each new season

# =============================================================================
# ENSEMBLE WEIGHTS — by season phase
# Format: (MC, Elo, RF, LR)
# =============================================================================
ENSEMBLE_WEIGHTS = {
    "opening":  (0.30, 0.15, 0.30, 0.25),  # Opening Day – Apr 15
    "early":    (0.45, 0.12, 0.25, 0.18),  # Apr 16 – May 31
    "mid":      (0.60, 0.10, 0.20, 0.10),  # June – July
    "late":     (0.65, 0.08, 0.18, 0.09),  # Aug – Sept
}

# Date boundaries for phase transitions
PHASE_EARLY_START   = (4, 16)   # (month, day)
PHASE_MID_START     = (6, 1)
PHASE_LATE_START    = (8, 1)

# =============================================================================
# BAYESIAN PRIOR (April early-season anchoring)
# =============================================================================
BAYESIAN_FULL_SEASON_PA    = 600        # PA threshold for 100% in-season weight
BAYESIAN_FULL_SEASON_BF    = 500        # BF threshold for pitchers
LOW_CONFIDENCE_PA_THRESH   = 50         # Below this: LOW CONFIDENCE flag on batter
LOW_CONFIDENCE_BF_THRESH   = 50         # Below this: LOW CONFIDENCE flag on pitcher

# =============================================================================
# EXPONENTIAL DECAY (recency weighting)
# =============================================================================
DECAY_FACTOR               = 0.92       # Weight = 0.92 ^ days_ago
DECAY_MIN_GAMES            = 3          # Min games needed before using decay stat
DECAY_WINDOW_DAYS          = 21         # Max days back for decay window

# =============================================================================
# PITCH-TYPE VULNERABILITY
# =============================================================================
PITCH_TYPE_MIN_PA          = 50         # Min PA vs. pitch type to use batter split
MATCHUP_MIN_PA             = 10         # Min historical PA for any matchup adjustment
MATCHUP_WEIGHT_SMALL       = 0.20       # Weight at 10-29 PA matchup history
MATCHUP_WEIGHT_MEDIUM      = 0.35       # Weight at 30-74 PA
MATCHUP_WEIGHT_LARGE       = 0.50       # Weight at 75+ PA

# =============================================================================
# TRUE SIERA FORMULA CONSTANTS (Zimmermann 2010)
# =============================================================================
SIERA_CONST                =  6.145
SIERA_K_COEF               = -16.986
SIERA_BB_COEF              =  11.434
SIERA_INTERACTION_COEF     = -1.858
SIERA_GB_FB_SQ_COEF        =  7.653
SIERA_LD_SQ_COEF           = -6.664
SIERA_GB_FB_K_COEF         =  10.130
SIERA_GB_FB_COEF           = -5.195

# =============================================================================
# TRUE-TALENT ERA BLEND WEIGHTS
# =============================================================================
ERA_BLEND_SIERA            = 0.45
ERA_BLEND_XERA             = 0.35
ERA_BLEND_FIP              = 0.20
FIP_CONSTANT               = 3.10       # 2024 MLB FIP constant

# =============================================================================
# LEAGUE BASELINES (2024 MLB averages)
# =============================================================================
LEAGUE_AVG_OBP             = 0.315
LEAGUE_AVG_SLG             = 0.413
LEAGUE_AVG_BABIP           = 0.296
LEAGUE_AVG_K_RATE          = 0.228
LEAGUE_AVG_BB_RATE         = 0.083
LEAGUE_AVG_HR_RATE         = 0.037      # HR per PA
LEAGUE_AVG_XWOBA           = 0.320
LEAGUE_AVG_BARREL_PCT      = 0.080
LEAGUE_AVG_SWSTR_PCT       = 0.115
LEAGUE_AVG_CSW_PCT         = 0.280
LEAGUE_AVG_SPRINT_SPEED    = 27.0       # ft/s
LEAGUE_AVG_ERA             = 4.20

# =============================================================================
# SPRINT SPEED TIERS (ft/s)
# =============================================================================
SPRINT_FAST_THRESHOLD      = 28.0
SPRINT_SLOW_THRESHOLD      = 25.0

# =============================================================================
# WEATHER COEFFICIENTS
# =============================================================================
# Wind run adjustments (runs/game added per speed bracket, directional)
WIND_OUT_BRACKETS = {
    (10, 14): +0.4,
    (15, 19): +0.7,
    (20, 24): +1.1,
    (25, 999): +1.5,
}
WIND_IN_BRACKETS = {
    (10, 14): -0.3,
    (15, 19): -0.6,
    (20, 24): -0.9,
    (25, 999): -1.2,
}
WIND_CROSS_ADJ             = 0.10       # Near-neutral crosswind
WIND_CALM_THRESHOLD        = 10         # mph — below this, wind is ignored

# Temperature run adjustments (runs/game vs. 70-79°F baseline)
TEMP_ADJUSTMENTS = {
    (-999, 49):  -0.4,
    (50,   59):  -0.2,
    (60,   69):  -0.1,
    (70,   79):   0.0,
    (80,   89):  +0.1,
    (90,   999): +0.2,
}

# =============================================================================
# DATA PIPELINE
# =============================================================================
STATCAST_CACHE_TTL_HOURS   = 23         # Refresh Statcast cache every 23 hours
LINEUP_FALLBACK_GAMES      = 5          # Games of history for projected batting order
EXTRA_INNINGS_THRESHOLD    = 10         # Prior-day innings count → bullpen fatigue flag
BULLPEN_FATIGUE_XFIP_ADJ   = 0.40      # Add this to xFIP if bullpen fatigued from prior game
MATCHUP_CACHE_REFRESH_DAYS = 7          # Refresh batter-vs-pitcher matchup data weekly
PARK_FACTOR_YEARS          = 3          # Years of Statcast data for park factor computation
SEPTEMBER_ROSTER_SIZE      = 28         # Expanded roster (vs. 26 regular season)
SEPTEMBER_BULLPEN_IP_BONUS = 0.5        # Extra IP available in September

# =============================================================================
# CALIBRATION
# =============================================================================
ISOTONIC_MIN_GAMES         = 1000       # Switch from Platt to isotonic after this many games
BRIER_ALERT_THRESHOLD      = 0.250      # Alert if 30-day Brier exceeds this

# =============================================================================
# EXTERNAL SERVICES
# =============================================================================
RESEND_API_KEY             = os.getenv("RESEND_API_KEY", "")
REPORT_EMAIL_TO            = os.getenv("REPORT_EMAIL_TO", "")
REPORT_EMAIL_FROM          = os.getenv("REPORT_EMAIL_FROM", "")
DISCORD_WEBHOOK_URL        = os.getenv("DISCORD_WEBHOOK_URL", "")
SENTRY_DSN                 = os.getenv("SENTRY_DSN", "")
