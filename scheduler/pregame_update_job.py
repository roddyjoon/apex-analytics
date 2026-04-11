"""
Apex Analytics — Pre-Game Update Pipeline Job
Runs at 1:00 PM PT (4:00 PM ET) daily.

Purpose: regenerate predictions with confirmed lineups where now available.
At 1 PM PT, most games have confirmed lineups posted (game time ~7 PM ET).

Differences from morning_job:
  - Re-fetches lineups via MLB Stats API (confirmed now)
  - Only re-simulates games where lineup changed vs. morning run
  - Skips games that have already started (< 2hr buffer)
  - Saves as report_type="pregame"
  - Does NOT update Elo (morning job handles that)

Returns:
  dict with:
    report_path    : str — path to generated pregame report
    updated_games  : int — number of games re-simulated with new lineups
    skipped_games  : int — games skipped (started or lineup unchanged)
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# How many hours before game time we stop re-simulating (game about to start)
GAME_START_BUFFER_HOURS = 2


def run_pregame_job(
    game_date:    Optional[date] = None,
    send_email:   bool = True,
    send_discord: bool = True,
    dry_run:      bool = False,
) -> dict:
    """
    Execute the pre-game update pipeline.

    Parameters
    ----------
    game_date    : Date to run for (defaults to today).
    send_email   : Send updated email report if lineups changed.
    send_discord : Post Discord update if lineups changed.
    dry_run      : Skip notifications and DB writes.

    Returns
    -------
    dict with report_path, updated_games, skipped_games, total_games.
    """
    if game_date is None:
        game_date = date.today()

    pipeline_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("APEX ANALYTICS — PRE-GAME UPDATE — %s", game_date.isoformat())
    logger.info("=" * 60)

    result = {"report_path": None, "updated_games": 0,
              "skipped_games": 0, "total_games": 0}

    # ── Fetch schedule ─────────────────────────────────────────────
    try:
        from data.ingestors.mlb_schedule import fetch_schedule
        games = fetch_schedule(game_date)
        games = [g for g in games if g.get("status") != "postponed"]
        result["total_games"] = len(games)
        logger.info("Pre-game update: %d games to check", len(games))
    except Exception as exc:
        logger.error("Schedule fetch failed: %s — aborting pre-game job.", exc)
        return result

    if not games:
        return result

    season = game_date.year
    games_data = []
    now_utc = datetime.now(timezone.utc)

    for game in games:
        game_pk   = game["game_pk"]
        home_abbr = game.get("home_team_abbr", "HOME")
        away_abbr = game.get("away_team_abbr", "AWAY")

        # ── Skip games that have already started ───────────────────
        if _game_already_started(game, now_utc):
            logger.info("  %s @ %s — SKIPPED (game in progress or completed)",
                        away_abbr, home_abbr)
            result["skipped_games"] += 1
            continue

        # ── Check if lineups changed since morning ─────────────────
        lineup_changed = _lineups_changed(game_pk, game_date)

        if not lineup_changed:
            logger.debug("  %s @ %s — lineup unchanged, re-using morning result",
                         away_abbr, home_abbr)
            # Still include in report — pull morning simulation from DB
            morning_data = _load_morning_result(game_pk, game_date)
            if morning_data:
                games_data.append(morning_data)
            result["skipped_games"] += 1
            continue

        # ── Lineup changed — re-simulate ───────────────────────────
        logger.info("  %s @ %s — lineup updated, re-simulating...",
                    away_abbr, home_abbr)
        game_start = time.perf_counter()

        try:
            from scheduler.morning_job import _process_single_game
            game_data = _process_single_game(
                game, game_date, season, report_type="pregame"
            )
            if game_data:
                games_data.append(game_data)
                result["updated_games"] += 1
                elapsed = time.perf_counter() - game_start
                home_pct = game_data["ensemble_result"].get("calibrated_prob", 0.5)
                logger.info(
                    "  Updated in %.1fs | %s %.1f%% — %s %.1f%%",
                    elapsed, home_abbr, home_pct * 100,
                    away_abbr, (1 - home_pct) * 100,
                )
        except Exception as exc:
            logger.error("  FAILED (%s @ %s): %s",
                         away_abbr, home_abbr, exc, exc_info=True)
            result["skipped_games"] += 1

    if not games_data:
        logger.info("No games to report on — all started or failed.")
        return result

    # ── Generate pre-game report ───────────────────────────────────
    try:
        from report.generator import generate_daily_report
        report_path = generate_daily_report(
            game_date=game_date,
            games_data=games_data,
            report_type="pregame",
        )
        result["report_path"] = report_path
        logger.info("Pre-game report generated: %s", report_path)
    except Exception as exc:
        logger.error("Pre-game report generation failed: %s", exc, exc_info=True)

    # ── Save updated results to DB ─────────────────────────────────
    if not dry_run and result["updated_games"] > 0:
        _save_updated_results(games_data, game_date)

    # ── Notifications (send whenever we have games to report on) ───
    if not dry_run and games_data and result["report_path"]:
        if send_email:
            _send_pregame_email(result["report_path"], game_date,
                                result["updated_games"], games_data)
        if send_discord:
            _send_pregame_discord(games_data, result["report_path"], game_date,
                                  result["updated_games"])

    elapsed = time.perf_counter() - pipeline_start
    logger.info(
        "PRE-GAME UPDATE COMPLETE — %d updated, %d skipped in %.1fs",
        result["updated_games"], result["skipped_games"], elapsed,
    )
    logger.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _game_already_started(game: dict, now_utc: datetime) -> bool:
    """
    Return True if the game has started or is within GAME_START_BUFFER_HOURS
    of its scheduled start time.
    """
    # Check status flag from schedule API
    status = game.get("status", "").lower()
    if status in ("in_progress", "final", "game_over", "completed"):
        return True

    # Check scheduled start time
    game_time_utc = game.get("game_time_utc")
    if game_time_utc:
        try:
            if isinstance(game_time_utc, str):
                from datetime import datetime as dt
                game_dt = dt.fromisoformat(game_time_utc.replace("Z", "+00:00"))
            else:
                game_dt = game_time_utc

            buffer = timedelta(hours=GAME_START_BUFFER_HOURS)
            if now_utc >= (game_dt - buffer):
                return True
        except Exception:
            pass

    return False


def _lineups_changed(game_pk: int, game_date: date) -> bool:
    """
    Compare current confirmed lineup (MLB Stats API) against the
    lineup stored from the morning run.

    Returns True if the confirmed lineup differs from morning's lineup
    (or if the morning run used a projected lineup and a confirmed one is now available).
    """
    try:
        from data.ingestors.mlb_lineups import fetch_confirmed_lineup
        from data.cache.db import get_session, Lineup

        # Check what was stored in the morning run
        with get_session() as session:
            morning_rows = session.query(Lineup).filter(
                Lineup.game_pk == game_pk,
                Lineup.report_type == "morning",
            ).all()

        # If nothing stored from morning, treat as changed (re-simulate)
        if not morning_rows:
            return True

        # If morning lineup was projected, try to get confirmed now
        morning_sources = {r.lineup_source for r in morning_rows}
        if "confirmed" not in morning_sources:
            # Try to fetch confirmed lineup now
            home_id = next((r.team_id for r in morning_rows), None)
            confirmed = fetch_confirmed_lineup(game_pk, "pregame") if home_id else {}
            if confirmed.get("lineup"):
                logger.debug("  game_pk=%d: confirmed lineup now available (was projected)",
                             game_pk)
                return True

        return False

    except Exception as exc:
        logger.debug("lineup_changed check failed for game_pk=%d: %s — re-simulating",
                     game_pk, exc)
        return True  # Conservative: re-simulate on uncertainty


def _load_morning_result(game_pk: int, game_date: date) -> Optional[dict]:
    """
    Load the morning simulation result from DB for a game that doesn't need re-simulation.
    Reconstructs a games_data-compatible dict using SimpleNamespace proxies so the
    email builder and report generator can render it without a full pipeline re-run.
    Returns None if not found.
    """
    import json
    from types import SimpleNamespace
    from datetime import datetime, timezone, timedelta

    try:
        from data.cache.db import get_session, SimulationResult, Game
        from sqlalchemy import text

        with get_session() as session:
            row = session.query(SimulationResult).filter(
                SimulationResult.game_pk == game_pk,
                SimulationResult.game_date == game_date.isoformat(),
                SimulationResult.report_type == "morning",
            ).first()

            if not row:
                logger.debug("  game_pk=%d: no morning result in DB", game_pk)
                return None

            # Load game metadata
            gr = session.query(Game).filter(Game.game_pk == game_pk).first()

            # Load lineup_cache for this game
            lc_rows = session.execute(
                text("SELECT team_abbr, lineup_json, is_confirmed "
                     "FROM lineup_cache WHERE game_pk=:pk AND report_type='morning'"),
                {"pk": game_pk},
            ).fetchall()

            # Load player stats for xwOBA lookup
            ps_rows = session.execute(
                text("SELECT player_id, stats_json FROM player_stats WHERE season=:yr"),
                {"yr": game_date.year},
            ).fetchall()

        # Build player stats lookup
        player_stats = {}
        for ps in ps_rows:
            try:
                player_stats[ps[0]] = json.loads(ps[1]) if ps[1] else {}
            except Exception:
                pass

        # Build lineup proxies
        def _build_lineup(team_abbr):
            for lc in lc_rows:
                if lc[0] == team_abbr:
                    slots = json.loads(lc[1]) if lc[1] else []
                    proxies = []
                    for slot in slots:
                        pid  = slot.get("player_id")
                        ps   = player_stats.get(pid, {}) if pid else {}
                        xwoba = ps.get("xwoba") or ps.get("estimated_woba_using_speedangle")
                        obp   = ps.get("obp") or ps.get("on_base_percent")
                        slg   = ps.get("slg") or ps.get("slg_percent")
                        proxies.append(SimpleNamespace(
                            batting_order = slot.get("batting_order", 9),
                            player_name   = slot.get("player_name", ""),
                            position      = slot.get("position", ""),
                            player_id     = pid,
                            xwoba         = xwoba,
                            obp           = obp,
                            slg           = slg,
                            is_confirmed  = slot.get("is_confirmed", False),
                        ))
                    return proxies
            return []

        home_abbr = gr.home_team_abbr if gr else "HOME"
        away_abbr = gr.away_team_abbr if gr else "AWAY"

        # Parse game time → ET label
        game_time_et = "TBD"
        if gr and gr.game_time_utc:
            try:
                dt = datetime.fromisoformat(str(gr.game_time_utc).replace("Z", "+00:00"))
                et = dt.astimezone(timezone(timedelta(hours=-4)))  # EDT
                game_time_et = et.strftime("%-I:%M %p")
            except Exception:
                pass

        ctx = SimpleNamespace(
            game_pk        = game_pk,
            home_team_abbr = home_abbr,
            away_team_abbr = away_abbr,
            home_starter   = None,
            away_starter   = None,
            home_lineup    = _build_lineup(home_abbr),
            away_lineup    = _build_lineup(away_abbr),
            park           = None,
        )

        cal_prob = row.calibrated_prob or 0.53
        result = {
            "game_context": ctx,
            "mc_result": {
                "home_win_pct":        row.home_win_pct,
                "projected_home_runs": row.projected_home_runs,
                "projected_away_runs": row.projected_away_runs,
                "projected_total":     row.projected_total,
                "n_iterations":        row.n_iterations or 7000,
            },
            "ensemble_result": {
                "mc_prob":         row.mc_prob,
                "elo_prob":        row.elo_prob,
                "rf_prob":         row.rf_prob,
                "lr_prob":         row.lr_prob,
                "raw_ensemble":    cal_prob,
                "calibrated_prob": cal_prob,
                "confidence_band": (max(0.35, cal_prob - 0.07), min(0.65, cal_prob + 0.07)),
            },
            "away_full":    away_abbr,
            "home_full":    home_abbr,
            "game_time_et": game_time_et,
            "stadium_name": gr.venue_name if gr else "",
        }
        logger.debug("  game_pk=%d: loaded morning result from DB (%s @ %s)", game_pk, away_abbr, home_abbr)
        return result

    except Exception as exc:
        logger.debug("Could not load morning result for game_pk=%d: %s", game_pk, exc)
        return None


def _save_updated_results(games_data: list, game_date: date) -> None:
    """Save re-simulated results to DB as pregame report type."""
    try:
        from data.cache.db import get_session, SimulationResult, CalibrationHistory
        with get_session() as session:
            for gd in games_data:
                ctx      = gd["game_context"]
                mc       = gd["mc_result"]
                ensemble = gd["ensemble_result"]

                sim_row = SimulationResult(
                    game_pk=ctx.game_pk,
                    game_date=game_date.isoformat(),
                    report_type="pregame",
                    home_win_pct=mc.get("home_win_pct", 0.53),
                    projected_home_runs=mc.get("projected_home_runs", 4.5),
                    projected_away_runs=mc.get("projected_away_runs", 4.2),
                    projected_total=mc.get("projected_total", 8.7),
                    n_iterations=mc.get("n_iterations", 7000),
                    mc_prob=ensemble.get("mc_prob", 0.53),
                    elo_prob=ensemble.get("elo_prob", 0.53),
                    rf_prob=ensemble.get("rf_prob", 0.53),
                    lr_prob=ensemble.get("lr_prob", 0.53),
                    ensemble_prob=ensemble.get("raw_ensemble", 0.53),
                    calibrated_prob=ensemble.get("calibrated_prob", 0.53),
                )
                session.merge(sim_row)

                # Update calibration record with more accurate pregame probability
                existing = session.query(CalibrationHistory).filter(
                    CalibrationHistory.game_pk == ctx.game_pk,
                ).first()
                if existing:
                    existing.ensemble_prob = ensemble.get("calibrated_prob", 0.53)
                else:
                    session.add(CalibrationHistory(
                        game_pk=ctx.game_pk,
                        game_date=game_date.isoformat(),
                        ensemble_prob=ensemble.get("calibrated_prob", 0.53),
                        actual_outcome=None,
                    ))

        logger.info("Saved %d pre-game results to DB.", len(games_data))
    except Exception as exc:
        logger.warning("DB save failed (non-critical): %s", exc)


def _send_pregame_email(
    report_path: str, game_date: date, n_updated: int, games_data: list = None
) -> None:
    try:
        from notification.email import send_report_email
        ok = send_report_email(
            report_path, game_date,
            games_data=games_data,
            subject_suffix=f"Pre-Game Update ({n_updated} lineups confirmed)"
        )
        if ok:
            logger.info("Pre-game email sent.")
    except Exception as exc:
        logger.warning("Pre-game email failed: %s", exc)


def _send_pregame_discord(
    games_data: list, report_path: str, game_date: date, n_updated: int
) -> None:
    try:
        from notification.discord import post_discord_summary
        ok = post_discord_summary(
            games_data, report_path, game_date,
            title=f"⚡ Pre-Game Update — {n_updated} Lineups Confirmed"
        )
        if ok:
            logger.info("Pre-game Discord summary posted.")
    except Exception as exc:
        logger.warning("Pre-game Discord post failed: %s", exc)
