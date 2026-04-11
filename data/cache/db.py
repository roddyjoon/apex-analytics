"""
Apex Analytics — SQLite DB Cache
Persistent storage for schedule, players, stats, Elo ratings,
calibration history, and simulation results.

DB path: $APEX_DB_PATH (default: /tmp/apex_analytics.db)

Design:
  - SQLAlchemy ORM with SQLite backend (no external DB required)
  - Tables auto-created on first import via init_db()
  - JSON blobs for complex nested data (stats, lineups, arsenals)
  - Simple get/upsert API — no raw SQL in callers
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator, Optional

from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Text, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# ── Engine setup ───────────────────────────────────────────────────────────────
_DB_PATH = os.environ.get("APEX_DB_PATH", "/tmp/apex_analytics.db")
_engine  = None
_Session = None


def _get_engine():
    global _engine, _Session
    if _engine is None:
        db_url = f"sqlite:///{_DB_PATH}"
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
        Base.metadata.create_all(_engine)
        logger.debug("DB initialized at %s", _DB_PATH)
    return _engine


# ── Base ───────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────────

class Game(Base):
    __tablename__ = "games"
    game_pk        = Column(Integer, primary_key=True)
    game_date      = Column(String, index=True)
    home_team_id   = Column(Integer)
    home_team_abbr = Column(String)
    away_team_id   = Column(Integer)
    away_team_abbr = Column(String)
    venue_id       = Column(Integer)
    venue_name     = Column(String)
    game_time_utc  = Column(String)
    status         = Column(String, default="scheduled")
    double_header  = Column(String, default="N")
    home_score     = Column(Integer)
    away_score     = Column(Integer)
    innings        = Column(Integer)
    home_win       = Column(Integer)    # 1=home won, 0=away won, None=not played


class Player(Base):
    __tablename__ = "players"
    player_id   = Column(Integer, primary_key=True)
    player_name = Column(String)
    team_id     = Column(Integer)
    position    = Column(String)
    bats        = Column(String)
    throws      = Column(String)
    updated_at  = Column(String)


class PlayerStats(Base):
    """Season stats (current + prior) stored as JSON blob."""
    __tablename__ = "player_stats"
    player_id  = Column(Integer, primary_key=True)
    season     = Column(Integer, primary_key=True)
    stat_type  = Column(String,  primary_key=True)  # "batter" | "pitcher"
    stats_json = Column(Text)   # main stats dict
    game_log_json = Column(Text)  # per-start/per-game log
    updated_at = Column(String)


class ParkFactor(Base):
    __tablename__ = "park_factors"
    team_abbr   = Column(String, primary_key=True)
    venue_id    = Column(Integer)
    venue_name  = Column(String)
    run_factor  = Column(Float, default=1.0)
    hr_factor   = Column(Float, default=1.0)
    updated_at  = Column(String)
    extra_json  = Column(Text)   # additional fields


class LineupCache(Base):
    """Lineup slot storage — one row per game/team/report_type."""
    __tablename__ = "lineup_cache"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    game_pk      = Column(Integer, index=True)
    team_id      = Column(Integer)
    team_abbr    = Column(String)
    report_type  = Column(String)   # "morning" | "pregame"
    lineup_json  = Column(Text)     # list of slot dicts
    is_confirmed = Column(Boolean, default=False)
    lineup_source = Column(String, default="projected")
    updated_at   = Column(String)


class Lineup(Base):
    """Alias used by pregame_update_job for lineup-change detection."""
    __tablename__ = "lineups"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    game_pk      = Column(Integer, index=True)
    team_id      = Column(Integer)
    report_type  = Column(String)
    lineup_source = Column(String, default="projected")
    updated_at   = Column(String)


class MatchupHistoryDB(Base):
    __tablename__ = "matchup_history"
    batter_id    = Column(Integer, primary_key=True)
    pitcher_id   = Column(Integer, primary_key=True)
    history_json = Column(Text)
    updated_at   = Column(String)


class PitcherArsenalDB(Base):
    __tablename__ = "pitcher_arsenal"
    pitcher_id   = Column(Integer, primary_key=True)
    season       = Column(Integer, primary_key=True)
    arsenal_json = Column(Text)
    updated_at   = Column(String)


class PitchTypeSplitsDB(Base):
    __tablename__ = "pitch_type_splits"
    player_id    = Column(Integer, primary_key=True)
    season       = Column(Integer, primary_key=True)
    splits_json  = Column(Text)
    updated_at   = Column(String)


class EloRating(Base):
    __tablename__ = "elo_ratings"
    team_id      = Column(Integer, primary_key=True)
    season       = Column(Integer, primary_key=True)
    team_abbr    = Column(String)
    elo          = Column(Float, default=1500.0)
    games_played = Column(Integer, default=0)
    updated_at   = Column(String)


class CalibrationHistory(Base):
    """
    One row per game. ensemble_prob filled at prediction time;
    actual_outcome (1=home won, 0=away won) filled post-game by Elo updater.
    """
    __tablename__ = "calibration_history"
    game_pk        = Column(Integer, primary_key=True)
    game_date      = Column(String, index=True)
    ensemble_prob  = Column(Float)
    actual_outcome = Column(Float)   # None until game is final


class SimulationResult(Base):
    """Full simulation output per game per report type."""
    __tablename__ = "simulation_results"
    game_pk             = Column(Integer, primary_key=True)
    game_date           = Column(String,  primary_key=True)
    report_type         = Column(String,  primary_key=True)
    home_win_pct        = Column(Float)
    projected_home_runs = Column(Float)
    projected_away_runs = Column(Float)
    projected_total     = Column(Float)
    n_iterations        = Column(Integer)
    mc_prob             = Column(Float)
    elo_prob            = Column(Float)
    rf_prob             = Column(Float)
    lr_prob             = Column(Float)
    ensemble_prob       = Column(Float)
    calibrated_prob     = Column(Float)
    updated_at          = Column(String)


class AccuracyLog(Base):
    """Season accuracy tracking — one row per game."""
    __tablename__ = "accuracy_log"
    id                 = Column(Integer, primary_key=True, autoincrement=True)
    game_pk            = Column(Integer, index=True)
    game_date          = Column(String, index=True)
    predicted_prob     = Column(Float)
    actual_outcome     = Column(Integer)   # 1=home won, 0=away won
    correct_prediction = Column(Boolean)
    updated_at         = Column(String)


# ── Session context manager ────────────────────────────────────────────────────

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session; auto-commits on success, rolls back on error."""
    _get_engine()
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Explicitly create all tables. Called from main.py on startup."""
    _get_engine()
    logger.info("DB tables ensured at %s", _DB_PATH)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


# ── Game ───────────────────────────────────────────────────────────────────────

def upsert_game(game_dict: dict) -> None:
    """Insert or update a game record."""
    try:
        _get_engine()
        game_date = game_dict.get("game_date")
        if isinstance(game_date, date):
            game_date = game_date.isoformat()

        game_time = game_dict.get("game_time_utc")
        if hasattr(game_time, "isoformat"):
            game_time = game_time.isoformat()

        with get_session() as session:
            row = session.get(Game, game_dict["game_pk"])
            if row is None:
                row = Game(game_pk=game_dict["game_pk"])
                session.add(row)
            row.game_date      = game_date
            row.home_team_id   = game_dict.get("home_team_id")
            row.home_team_abbr = game_dict.get("home_team_abbr", "")
            row.away_team_id   = game_dict.get("away_team_id")
            row.away_team_abbr = game_dict.get("away_team_abbr", "")
            row.venue_id       = game_dict.get("venue_id")
            row.venue_name     = game_dict.get("venue_name", "")
            row.game_time_utc  = game_time
            row.status         = game_dict.get("status", "scheduled")
            row.double_header  = game_dict.get("double_header", "N")
            row.home_score     = game_dict.get("home_score")
            row.away_score     = game_dict.get("away_score")
            row.innings        = game_dict.get("innings")
            row.home_win       = game_dict.get("home_win")
    except Exception as exc:
        logger.debug("upsert_game failed for game_pk=%s: %s", game_dict.get("game_pk"), exc)


def get_games_for_date(game_date) -> list:
    """Return all games for a given date from DB. Returns [] if none found."""
    try:
        _get_engine()
        if isinstance(game_date, date):
            game_date = game_date.isoformat()
        with get_session() as session:
            rows = session.query(Game).filter(Game.game_date == game_date).all()
        result = []
        for row in rows:
            result.append({
                "game_pk":        row.game_pk,
                "game_date":      row.game_date,
                "home_team_id":   row.home_team_id,
                "home_team_abbr": row.home_team_abbr,
                "away_team_id":   row.away_team_id,
                "away_team_abbr": row.away_team_abbr,
                "venue_id":       row.venue_id,
                "venue_name":     row.venue_name,
                "status":         row.status,
                "double_header":  row.double_header,
                "home_score":     row.home_score,
                "away_score":     row.away_score,
                "home_win":       row.home_win,
            })
        return result
    except Exception as exc:
        logger.debug("get_games_for_date failed: %s", exc)
        return []


# ── Player ─────────────────────────────────────────────────────────────────────

def upsert_player(player_dict: dict) -> None:
    try:
        _get_engine()
        pid = player_dict.get("player_id")
        if not pid:
            return
        with get_session() as session:
            row = session.get(Player, pid)
            if row is None:
                row = Player(player_id=pid)
                session.add(row)
            row.player_name = player_dict.get("player_name", "")
            row.team_id     = player_dict.get("team_id")
            row.position    = player_dict.get("position", "")
            row.bats        = player_dict.get("bats", "")
            row.throws      = player_dict.get("throws", "")
            row.updated_at  = _now()
    except Exception as exc:
        logger.debug("upsert_player failed: %s", exc)


def get_player(player_id: int) -> Optional[dict]:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(Player, player_id)
        if row is None:
            return None
        return {
            "player_id":   row.player_id,
            "player_name": row.player_name,
            "team_id":     row.team_id,
            "position":    row.position,
            "bats":        row.bats,
            "throws":      row.throws,
        }
    except Exception as exc:
        logger.debug("get_player failed: %s", exc)
        return None


# ── Player Stats ───────────────────────────────────────────────────────────────

def save_player_stats(
    player_id: int,
    season: int,
    stat_type: str,
    stats: dict,
    game_log: Optional[Any] = None,
) -> None:
    """Persist season stats dict (and optional game log) to DB."""
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(PlayerStats, (player_id, season, stat_type))
            if row is None:
                row = PlayerStats(
                    player_id=player_id, season=season, stat_type=stat_type
                )
                session.add(row)
            row.stats_json    = json.dumps(stats,    default=str)
            row.game_log_json = json.dumps(game_log, default=str) if game_log is not None else None
            row.updated_at    = _now()
    except Exception as exc:
        logger.debug("save_player_stats failed for player_id=%d: %s", player_id, exc)


def get_player_stats(player_id: int, season: int, stat_type: str) -> Optional[dict]:
    """
    Return stored stats as {"stats": {...}, "game_log": [...]} or None if not found.
    The "stats" value is the flat stats dict; "game_log" is a list (may be empty).
    """
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(PlayerStats, (player_id, season, stat_type))
        if row is None:
            return None
        stats    = json.loads(row.stats_json)    if row.stats_json    else {}
        game_log = json.loads(row.game_log_json) if row.game_log_json else []
        return {"stats": stats, "game_log": game_log}
    except Exception as exc:
        logger.debug("get_player_stats failed: %s", exc)
        return None


def get_prior_stats(player_id: int, season: int, stat_type: str) -> dict:
    """
    Return stored stats for a given player/season/type.
    Used by bayesian_prior for prior-season anchoring.
    Returns {} (not None) so callers can safely .get() on result.
    """
    wrapped = get_player_stats(player_id, season, stat_type)
    if wrapped is None:
        return {}
    result = dict(wrapped.get("stats", {}))
    result["_source"] = "db"
    return result


# ── Park Factors ───────────────────────────────────────────────────────────────

def save_park_factor(venue_id_or_abbr: Any, data: dict) -> None:
    """
    Save park factor. Keyed by team_abbr (extracted from data dict).
    venue_id_or_abbr can be an int venue_id or string team_abbr — we prefer data["team_abbr"].
    """
    try:
        _get_engine()
        abbr = data.get("team_abbr") or str(venue_id_or_abbr)
        if not abbr:
            return
        with get_session() as session:
            row = session.get(ParkFactor, abbr)
            if row is None:
                row = ParkFactor(team_abbr=abbr)
                session.add(row)
            row.venue_id   = data.get("venue_id") or (
                int(venue_id_or_abbr) if isinstance(venue_id_or_abbr, int) else 0
            )
            row.venue_name = data.get("venue_name", "")
            row.run_factor = float(data.get("run_factor", 1.0))
            row.hr_factor  = float(data.get("hr_factor", 1.0))
            row.updated_at = _now()
            # Store remaining keys as JSON
            extra = {k: v for k, v in data.items()
                     if k not in ("team_abbr", "venue_id", "venue_name", "run_factor", "hr_factor")}
            row.extra_json = json.dumps(extra) if extra else None
    except Exception as exc:
        logger.debug("save_park_factor failed: %s", exc)


def get_park_factor(team_abbr: str) -> Optional[dict]:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(ParkFactor, team_abbr.upper())
        if row is None:
            return None
        return {
            "team_abbr":  row.team_abbr,
            "venue_name": row.venue_name,
            "run_factor": row.run_factor,
            "hr_factor":  row.hr_factor,
        }
    except Exception as exc:
        logger.debug("get_park_factor failed: %s", exc)
        return None


# ── Lineups ────────────────────────────────────────────────────────────────────

def save_lineups(
    game_pk: int,
    team_id: int,
    team_abbr: str,
    lineup: list,
    report_type: str = "morning",
) -> None:
    try:
        _get_engine()
        is_confirmed  = any(s.get("is_confirmed") for s in lineup) if lineup else False
        lineup_source = "confirmed" if is_confirmed else "projected"
        with get_session() as session:
            # Upsert LineupCache
            existing = session.query(LineupCache).filter(
                LineupCache.game_pk == game_pk,
                LineupCache.team_id == team_id,
                LineupCache.report_type == report_type,
            ).first()
            if existing is None:
                existing = LineupCache(
                    game_pk=game_pk, team_id=team_id,
                    team_abbr=team_abbr, report_type=report_type
                )
                session.add(existing)
            existing.lineup_json  = json.dumps(lineup, default=str)
            existing.is_confirmed = is_confirmed
            existing.lineup_source = lineup_source
            existing.updated_at   = _now()

            # Also write to Lineup table (used by pregame_update_job)
            lineup_row = session.query(Lineup).filter(
                Lineup.game_pk == game_pk,
                Lineup.team_id == team_id,
                Lineup.report_type == report_type,
            ).first()
            if lineup_row is None:
                lineup_row = Lineup(
                    game_pk=game_pk, team_id=team_id, report_type=report_type
                )
                session.add(lineup_row)
            lineup_row.lineup_source = lineup_source
            lineup_row.updated_at    = _now()
    except Exception as exc:
        logger.debug("save_lineups failed for game_pk=%d: %s", game_pk, exc)


def get_lineups(game_pk: int, team_id: int, report_type: str = "morning") -> Optional[list]:
    try:
        _get_engine()
        with get_session() as session:
            row = session.query(LineupCache).filter(
                LineupCache.game_pk == game_pk,
                LineupCache.team_id == team_id,
                LineupCache.report_type == report_type,
            ).first()
        if row is None:
            return None
        return json.loads(row.lineup_json) if row.lineup_json else []
    except Exception as exc:
        logger.debug("get_lineups failed: %s", exc)
        return None


# ── Matchup History ────────────────────────────────────────────────────────────

def save_matchup_history(batter_id: int, pitcher_id: int, history: Any) -> None:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(MatchupHistoryDB, (batter_id, pitcher_id))
            if row is None:
                row = MatchupHistoryDB(batter_id=batter_id, pitcher_id=pitcher_id)
                session.add(row)
            row.history_json = json.dumps(history, default=str)
            row.updated_at   = _now()
    except Exception as exc:
        logger.debug("save_matchup_history failed: %s", exc)


def get_matchup_history(batter_id: int, pitcher_id: int) -> Optional[dict]:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(MatchupHistoryDB, (batter_id, pitcher_id))
        if row is None:
            return None
        return json.loads(row.history_json) if row.history_json else None
    except Exception as exc:
        logger.debug("get_matchup_history failed: %s", exc)
        return None


# ── Pitcher Arsenal ────────────────────────────────────────────────────────────

def save_pitcher_arsenal(pitcher_id: int, season: int, arsenal: Any) -> None:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(PitcherArsenalDB, (pitcher_id, season))
            if row is None:
                row = PitcherArsenalDB(pitcher_id=pitcher_id, season=season)
                session.add(row)
            row.arsenal_json = json.dumps(arsenal, default=str)
            row.updated_at   = _now()
    except Exception as exc:
        logger.debug("save_pitcher_arsenal failed: %s", exc)


def get_pitcher_arsenal(pitcher_id: int, season: int) -> Optional[list]:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(PitcherArsenalDB, (pitcher_id, season))
        if row is None:
            return None
        return json.loads(row.arsenal_json) if row.arsenal_json else []
    except Exception as exc:
        logger.debug("get_pitcher_arsenal failed: %s", exc)
        return None


# ── Pitch Type Splits ──────────────────────────────────────────────────────────

def save_pitch_type_splits(player_id: int, season: int, splits: Any) -> None:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(PitchTypeSplitsDB, (player_id, season))
            if row is None:
                row = PitchTypeSplitsDB(player_id=player_id, season=season)
                session.add(row)
            row.splits_json = json.dumps(splits, default=str)
            row.updated_at  = _now()
    except Exception as exc:
        logger.debug("save_pitch_type_splits failed: %s", exc)


def get_pitch_type_splits(player_id: int, season: int) -> Optional[dict]:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(PitchTypeSplitsDB, (player_id, season))
        if row is None:
            return None
        return json.loads(row.splits_json) if row.splits_json else None
    except Exception as exc:
        logger.debug("get_pitch_type_splits failed: %s", exc)
        return None


# ── Elo Ratings ────────────────────────────────────────────────────────────────

def upsert_elo(
    team_id:      int,
    team_abbr:    str,
    season:       int,
    elo:          float,
    games_played: int = 0,
) -> None:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(EloRating, (team_id, season))
            if row is None:
                row = EloRating(team_id=team_id, season=season)
                session.add(row)
            row.team_abbr    = team_abbr
            row.elo          = float(elo)
            row.games_played = games_played
            row.updated_at   = _now()
    except Exception as exc:
        logger.debug("upsert_elo failed for team_id=%d: %s", team_id, exc)


def get_elo(team_id: int, season: int) -> Optional[float]:
    try:
        _get_engine()
        with get_session() as session:
            row = session.get(EloRating, (team_id, season))
        return float(row.elo) if row else None
    except Exception as exc:
        logger.debug("get_elo failed: %s", exc)
        return None
