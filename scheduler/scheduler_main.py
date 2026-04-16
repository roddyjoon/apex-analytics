"""
Apex Analytics — Persistent Scheduler
Runs morning job at 8:00 AM PT, pre-game update at 1:00 PM PT,
and end-of-day results at 11:00 PM PT daily.
"""
import logging
import os
from datetime import date, datetime, timezone, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-30s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex.scheduler")

PT = pytz.timezone("America/Los_Angeles")


def _pt_date() -> date:
    """Return today's date in Pacific Time (works correctly on UTC servers)."""
    return datetime.now(PT).date()


def morning_job():
    from scheduler.morning_job import run_morning_job
    today = _pt_date()
    logger.info("Firing morning job for %s", today)
    run_morning_job(game_date=today, send_email=True, send_discord=True)


def pregame_job():
    from scheduler.pregame_update_job import run_pregame_job
    today = _pt_date()
    logger.info("Firing pre-game update for %s", today)
    run_pregame_job(game_date=today, send_email=True, send_discord=True)


def results_job():
    from scheduler.results_job import run_results_job
    today = _pt_date()
    logger.info("Firing end-of-day results for %s", today)
    result = run_results_job(game_date=today, send_email=True, send_discord=False)
    logger.info(
        "Results complete: %d final, %d/%d correct (%.1f%%), Brier=%.4f, email=%s",
        result["n_final"], result["n_correct"], result["n_final"],
        result["accuracy"] * 100, result["brier_score"],
        "sent" if result["email_sent"] else "FAILED",
    )


def main():
    """Entry point — works both as a script and via `python -m scheduler.scheduler_main`."""
    from dotenv import load_dotenv
    load_dotenv()

    scheduler = BlockingScheduler(timezone=PT)

    # Morning report: 8:00 AM PT daily
    scheduler.add_job(
        morning_job,
        CronTrigger(hour=8, minute=0, timezone=PT),
        id="morning_report",
        name="Morning Report (8 AM PT)",
        max_instances=1,
        misfire_grace_time=300,
    )

    # Pre-game update: 1:00 PM PT daily
    scheduler.add_job(
        pregame_job,
        CronTrigger(hour=13, minute=0, timezone=PT),
        id="pregame_update",
        name="Pre-Game Update (1 PM PT)",
        max_instances=1,
        misfire_grace_time=300,
    )

    # End-of-day results: 11:00 PM PT daily
    scheduler.add_job(
        results_job,
        CronTrigger(hour=23, minute=0, timezone=PT),
        id="end_of_day_results",
        name="End-of-Day Results (11 PM PT)",
        max_instances=1,
        misfire_grace_time=3600,
    )

    logger.info("Scheduler started. Jobs:")
    for job in scheduler.get_jobs():
        logger.info("  %s (id=%s)", job.name, job.id)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
