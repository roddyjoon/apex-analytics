"""
Apex Analytics — APScheduler Entry Point
Runs the two daily pipeline jobs on a cron schedule.

Jobs:
  Morning job:   08:00 AM PT (11:00 AM ET) — projected lineups
  Pre-game job:  01:00 PM PT (04:00 PM ET) — confirmed lineups
  Elo update:    02:00 AM PT (05:00 AM ET) — post-game Elo update

Start with:
  python -m scheduler.main
  python main.py --scheduler

Stops cleanly on SIGTERM / SIGINT (Ctrl+C).
"""

import logging
import signal
import sys
import time
from datetime import date

logger = logging.getLogger(__name__)


def start_scheduler(dry_run: bool = False) -> None:
    """
    Start APScheduler with all three cron jobs and block until shutdown.

    Parameters
    ----------
    dry_run : If True, run jobs in test mode (no notifications, no DB writes).
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.executors.pool import ThreadPoolExecutor
    except ImportError:
        logger.error(
            "APScheduler not installed. Run: pip install apscheduler\n"
            "Then restart the scheduler."
        )
        sys.exit(1)

    executors = {"default": ThreadPoolExecutor(max_workers=2)}
    scheduler = BackgroundScheduler(
        executors=executors,
        timezone="America/Los_Angeles",
        job_defaults={"coalesce": True, "max_instances": 1},
    )

    # ── Morning Report — 8:00 AM PT ───────────────────────────────
    scheduler.add_job(
        func=_run_morning,
        trigger="cron",
        hour=8,
        minute=0,
        id="morning_report",
        name="Morning Report (Projected Lineups)",
        kwargs={"dry_run": dry_run},
        replace_existing=True,
    )

    # ── Pre-Game Update — 1:00 PM PT ──────────────────────────────
    scheduler.add_job(
        func=_run_pregame,
        trigger="cron",
        hour=13,
        minute=0,
        id="pregame_update",
        name="Pre-Game Update (Confirmed Lineups)",
        kwargs={"dry_run": dry_run},
        replace_existing=True,
    )

    # ── Elo Update — 2:00 AM PT (after all games end) ─────────────
    scheduler.add_job(
        func=_run_elo_update,
        trigger="cron",
        hour=2,
        minute=0,
        id="elo_update",
        name="Nightly Elo Rating Update",
        kwargs={"dry_run": dry_run},
        replace_existing=True,
    )

    # ── End-of-Day Results — 11:00 PM PT ─────────────────────────
    scheduler.add_job(
        func=_run_results,
        trigger="cron",
        hour=23,
        minute=0,
        id="end_of_day_results",
        name="End-of-Day Results & Accuracy Report",
        kwargs={"dry_run": dry_run},
        replace_existing=True,
    )

    # ── Calibration check — 3:00 AM PT (weekly, Monday only) ──────
    scheduler.add_job(
        func=_run_calibration_check,
        trigger="cron",
        day_of_week="mon",
        hour=3,
        minute=0,
        id="calibration_check",
        name="Weekly Calibration Health Check",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("=" * 60)
    logger.info("Apex Analytics Scheduler STARTED")
    logger.info("  Morning report:  08:00 AM PT daily")
    logger.info("  Pre-game update: 01:00 PM PT daily")
    logger.info("  End-of-day:      11:00 PM PT daily")
    logger.info("  Elo update:      02:00 AM PT daily")
    logger.info("  Calibration:     03:00 AM PT Mondays")
    logger.info("  Mode: %s", "DRY RUN" if dry_run else "PRODUCTION")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 60)

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping scheduler...")
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # Block main thread
    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Keyboard interrupt — stopping scheduler.")
        scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Job wrapper functions (APScheduler calls these in threads)
# ---------------------------------------------------------------------------

def _run_morning(dry_run: bool = False) -> None:
    """Morning job wrapper with top-level exception guard."""
    try:
        logger.info("SCHEDULED: Starting morning report job...")
        from scheduler.morning_job import run_morning_job
        report_path = run_morning_job(
            game_date=date.today(),
            send_email=True,
            send_discord=True,
            dry_run=dry_run,
        )
        if report_path:
            logger.info("Morning job COMPLETE — report: %s", report_path)
        else:
            logger.warning("Morning job completed with no report generated.")
    except Exception as exc:
        logger.error("Morning job CRASHED: %s", exc, exc_info=True)
        _send_failure_alert("morning_report", exc)


def _run_pregame(dry_run: bool = False) -> None:
    """Pre-game update wrapper with top-level exception guard."""
    try:
        logger.info("SCHEDULED: Starting pre-game update job...")
        from scheduler.pregame_update_job import run_pregame_job
        result = run_pregame_job(
            game_date=date.today(),
            send_email=True,
            send_discord=True,
            dry_run=dry_run,
        )
        logger.info(
            "Pre-game job COMPLETE — updated=%d skipped=%d",
            result.get("updated_games", 0), result.get("skipped_games", 0),
        )
    except Exception as exc:
        logger.error("Pre-game job CRASHED: %s", exc, exc_info=True)
        _send_failure_alert("pregame_update", exc)


def _run_elo_update(dry_run: bool = False) -> None:
    """
    Nightly Elo update: fetch yesterday's final results and update ratings.
    Also fills in actual_outcome for CalibrationHistory records.
    """
    try:
        from datetime import timedelta
        yesterday = date.today() - timedelta(days=1)
        logger.info("SCHEDULED: Nightly Elo update for %s...", yesterday.isoformat())

        from data.ingestors.mlb_results import batch_fetch_results
        from ensemble.elo_system import batch_update_from_results
        results = batch_fetch_results(yesterday)

        if not results:
            logger.info("No results found for %s.", yesterday.isoformat())
            return

        if not dry_run:
            updates = batch_update_from_results(results, yesterday.year)
            logger.info("Elo updated for %d games.", len(updates))
            _update_calibration_outcomes(results, yesterday)
        else:
            logger.info("DRY RUN: would update Elo for %d games.", len(results))

    except Exception as exc:
        logger.error("Elo update job CRASHED: %s", exc, exc_info=True)


def _run_results(dry_run: bool = False) -> None:
    """End-of-day results job wrapper."""
    try:
        logger.info("SCHEDULED: Starting end-of-day results job...")
        # APScheduler fires at 11 PM PT, but Railway runs UTC (7 AM next day).
        # Use PT timezone explicitly so we grade the correct date's games.
        from datetime import datetime, timezone, timedelta
        pt_now = datetime.now(timezone(timedelta(hours=-7)))  # PDT = UTC-7
        game_date_pt = pt_now.date()
        # If it's past midnight PT but we're grading last night's games, go back 1 day
        if pt_now.hour < 4:
            game_date_pt = game_date_pt - timedelta(days=1)
        logger.info("Grading games for %s (PT time: %s)", game_date_pt, pt_now.strftime("%H:%M"))

        from scheduler.results_job import run_results_job
        result = run_results_job(
            game_date=game_date_pt,
            send_email=True,
            send_discord=False,
            dry_run=dry_run,
        )
        logger.info(
            "Results job complete: %d final, %d/%d correct (%.1f%%), Brier=%.4f",
            result["n_final"], result["n_correct"], result["n_final"],
            result["accuracy"] * 100, result["brier_score"],
        )
    except Exception as exc:
        logger.error("Results job CRASHED: %s", exc, exc_info=True)


def _run_calibration_check() -> None:
    """Weekly calibration health check — logs results, alerts if Brier degrades."""
    try:
        logger.info("SCHEDULED: Weekly calibration health check...")
        from ensemble.calibrator import check_calibration_health, fit_from_db
        health = check_calibration_health(window_days=30)
        logger.info(
            "Calibration health: status=%s Brier=%.4f accuracy=%.3f n=%d",
            health.get("status", "?"), health.get("brier_calibrated", 0),
            health.get("accuracy", 0), health.get("n_games", 0),
        )
        # Refit calibrator from full DB history
        metrics = fit_from_db()
        if metrics:
            logger.info(
                "Calibrator refit: method=%s n=%d Brier=%.4f",
                metrics.get("method"), metrics.get("n_games"), metrics.get("brier_score", 0),
            )
    except Exception as exc:
        logger.error("Calibration check CRASHED: %s", exc, exc_info=True)


def _update_calibration_outcomes(results: list, game_date) -> None:
    """
    Fill in actual_outcome (1=home won, 0=home lost) for CalibrationHistory records
    after last night's games are final.
    """
    try:
        from data.cache.db import get_session, CalibrationHistory
        with get_session() as session:
            for r in results:
                game_pk  = r.get("game_pk")
                home_won = r.get("home_win")
                if game_pk is None or home_won is None:
                    continue

                row = session.query(CalibrationHistory).filter(
                    CalibrationHistory.game_pk == game_pk,
                ).first()
                if row:
                    row.actual_outcome = 1 if home_won else 0

        logger.info("Calibration outcomes updated for %d games.", len(results))
    except Exception as exc:
        logger.warning("Could not update calibration outcomes: %s", exc)


def _send_failure_alert(job_name: str, exc: Exception) -> None:
    """
    Send a failure alert via Discord if configured.
    Non-critical — failures here are logged and swallowed.
    """
    try:
        import os
        import requests
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            return
        requests.post(webhook_url, json={
            "content": (
                f"⚠️ **Apex Analytics Pipeline Failure**\n"
                f"Job: `{job_name}`\n"
                f"Error: `{type(exc).__name__}: {str(exc)[:200]}`"
            )
        }, timeout=5)
    except Exception:
        pass  # Never let the alert crash the scheduler


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    start_scheduler()
