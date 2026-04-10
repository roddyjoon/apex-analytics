"""
Apex Analytics — CLI Entry Point
Educational MLB win-probability prediction engine.

Usage:
  python main.py                              # Run full morning pipeline for today
  python main.py --report-today              # Alias for full morning pipeline
  python main.py --date 2025-04-10           # Run pipeline for a specific date
  python main.py --fetch-only                # Fetch data only (no simulation)
  python main.py --fetch-only --date 2025-04-10
  python main.py --game-pk 745502            # Fetch one game (with --fetch-only)
  python main.py --simulate --game-pk 748532 # Run MC simulation for one game
  python main.py --report --date 2025-04-10  # Generate report from cached data
  python main.py --update-elo --date 2025-04-10  # Update Elo from that date's results
  python main.py --backtest-date 2024-08-15  # Run full pipeline on historical date
  python main.py --scheduler                 # Start the APScheduler background process
  python main.py --dry-run --report-today    # Run pipeline without notifications/DB writes
"""

import argparse
import logging
import sys
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Configure logging before any other imports so all modules inherit it.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="apex-analytics",
        description="Apex Analytics — Educational MLB win probability engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Date override (used by most commands)
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Override today's date for data fetch/report (default: today).",
    )
    parser.add_argument(
        "--game-pk",
        metavar="GAMEPK",
        type=int,
        default=None,
        help="Filter to a single game by MLB game PK.",
    )

    # ── Pipeline modes (mutually exclusive) ──────────────────────
    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--fetch-only",
        action="store_true",
        help="Run data fetch pipeline only (no simulation, no report).",
    )
    mode.add_argument(
        "--simulate",
        action="store_true",
        help="Run Monte Carlo simulation for one game (requires --game-pk).",
    )
    mode.add_argument(
        "--report-today",
        action="store_true",
        help="Run full morning pipeline for today (fetch + sim + report).",
    )
    mode.add_argument(
        "--report",
        action="store_true",
        help="Generate report for --date (uses cached simulation data).",
    )
    mode.add_argument(
        "--update-elo",
        action="store_true",
        help="Update Elo ratings from results on --date.",
    )
    mode.add_argument(
        "--backtest-date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Run full pipeline on a historical date (for testing/validation).",
    )
    mode.add_argument(
        "--scheduler",
        action="store_true",
        help="Start the APScheduler background process (production mode).",
    )

    # ── Options ──────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip notifications and DB writes (testing mode).",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Skip email notification even if RESEND_API_KEY is set.",
    )
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="Skip Discord notification even if DISCORD_WEBHOOK_URL is set.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG level logging.",
    )

    return parser.parse_args()


def _parse_date(date_str: Optional[str]) -> date:
    if date_str is None:
        return date.today()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.error("Invalid date format: %r — expected YYYY-MM-DD.", date_str)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_fetch_only(game_date: date, game_pk_filter: Optional[int]) -> None:
    """Run the data fetch pipeline and print a summary table."""
    logger.info("Starting fetch-only pipeline for %s...", game_date.isoformat())

    try:
        from data.ingestors.mlb_schedule import fetch_schedule
        games = fetch_schedule(game_date)
    except Exception as exc:
        logger.error("Schedule fetch failed: %s", exc)
        sys.exit(1)

    if game_pk_filter:
        games = [g for g in games if g.get("game_pk") == game_pk_filter]
        if not games:
            logger.warning("No game found with game_pk=%d on %s.",
                           game_pk_filter, game_date.isoformat())
            return

    if not games:
        logger.info("No games scheduled for %s.", game_date.isoformat())
        return

    logger.info("Found %d games. Fetching data...", len(games))
    rows = []

    for game in games:
        game_pk   = game["game_pk"]
        home_abbr = game.get("home_team_abbr", "HOME")
        away_abbr = game.get("away_team_abbr", "AWAY")
        row = {
            "game_pk": game_pk,
            "matchup": f"{away_abbr} @ {home_abbr}",
            "home_pitcher_bf": None, "away_pitcher_bf": None,
            "home_pitcher_conf": "?", "away_pitcher_conf": "?",
            "home_roster_size": None, "away_roster_size": None,
            "weather_note": "—", "errors": [],
        }

        # Probable pitchers
        try:
            from data.ingestors.mlb_lineups import fetch_probable_pitchers
            pitchers = fetch_probable_pitchers(game_pk)
            if pitchers.get("home_pitcher_id"):
                from data.ingestors.statcast_pitcher import fetch_pitcher_stats
                stats = fetch_pitcher_stats(
                    pitchers["home_pitcher_id"], game_date.year,
                    team_id=game.get("home_team_id", 0)
                )
                row["home_pitcher_bf"]   = stats.get("bf", "?")
                row["home_pitcher_conf"] = stats.get("confidence", "LOW")
            if pitchers.get("away_pitcher_id"):
                from data.ingestors.statcast_pitcher import fetch_pitcher_stats
                stats = fetch_pitcher_stats(
                    pitchers["away_pitcher_id"], game_date.year,
                    team_id=game.get("away_team_id", 0)
                )
                row["away_pitcher_bf"]   = stats.get("bf", "?")
                row["away_pitcher_conf"] = stats.get("confidence", "LOW")
        except Exception as exc:
            row["errors"].append(f"pitcher: {exc}")

        # Roster sizes
        try:
            from data.ingestors.mlb_rosters import fetch_roster
            home_roster = fetch_roster(game.get("home_team_id", 0), game_date)
            away_roster = fetch_roster(game.get("away_team_id", 0), game_date)
            row["home_roster_size"] = len(home_roster)
            row["away_roster_size"] = len(away_roster)
        except Exception as exc:
            row["errors"].append(f"roster: {exc}")

        # Weather
        try:
            from data.ingestors.weather import fetch_weather
            from data.ingestors.mlb_schedule import get_venue_coords_by_team
            coords = get_venue_coords_by_team(game.get("home_team_abbr", ""))
            if coords:
                wx = fetch_weather(coords["lat"], coords["lon"], game_date)
                row["weather_note"] = (
                    f"{wx.get('temperature_f', '?')}°F, "
                    f"wind {wx.get('wind_speed_mph', 0):.0f}mph "
                    f"{wx.get('wind_direction_label', '')}"
                )
        except Exception as exc:
            row["errors"].append(f"weather: {exc}")

        rows.append(row)
        logger.info("  %s: fetched", row["matchup"])

    _print_fetch_summary(rows, game_date)


def cmd_simulate(game_pk: int, game_date: date) -> None:
    """Run Monte Carlo simulation for one game and print results."""
    if not game_pk:
        logger.error("--simulate requires --game-pk GAMEPK")
        sys.exit(1)

    logger.info("Running simulation for game_pk=%d on %s...", game_pk, game_date.isoformat())

    try:
        from data.ingestors.mlb_schedule import fetch_schedule
        games = fetch_schedule(game_date)
        game = next((g for g in games if g["game_pk"] == game_pk), None)
        if not game:
            logger.error("game_pk=%d not found in schedule for %s.", game_pk, game_date.isoformat())
            sys.exit(1)

        from scheduler.morning_job import _process_single_game
        game_data = _process_single_game(game, game_date, game_date.year, "morning")

        if game_data:
            mc  = game_data["mc_result"]
            ens = game_data["ensemble_result"]
            ctx = game_data["game_context"]
            print(f"\n{'='*55}")
            print(f"  {ctx.away_team_abbr} @ {ctx.home_team_abbr}  |  game_pk={game_pk}")
            print(f"{'='*55}")
            print(f"  Monte Carlo:  {mc['home_win_pct']*100:.1f}% home  ({mc['n_iterations']:,} iterations, {mc['elapsed_seconds']:.1f}s)")
            print(f"  Ensemble:     {ens['calibrated_prob']*100:.1f}% home  (calibrated)")
            print(f"    MC={ens['mc_prob']*100:.1f}%  Elo={ens['elo_prob']*100:.1f}%  RF={ens['rf_prob']*100:.1f}%  LR={ens['lr_prob']*100:.1f}%")
            print(f"  Proj. total:  {mc['projected_total']:.1f} runs")
            print(f"  Extra inn.:   {mc['extra_innings_pct']*100:.1f}%")
            print(f"  CI band:      {ens['confidence_band'][0]*100:.1f}% – {ens['confidence_band'][1]*100:.1f}%")
            rd = mc["run_distribution"]
            print(f"  Run dist:     P5={rd[5]} P25={rd[25]} Med={rd[50]} P75={rd[75]} P95={rd[95]}")
            print(f"{'='*55}\n")
        else:
            logger.error("Simulation returned no result for game_pk=%d.", game_pk)

    except Exception as exc:
        logger.error("Simulation failed: %s", exc, exc_info=True)
        sys.exit(1)


def cmd_report_today(game_date: date, dry_run: bool,
                     no_email: bool, no_discord: bool) -> None:
    """Run the full morning pipeline."""
    logger.info("Running full morning pipeline for %s...", game_date.isoformat())
    try:
        from scheduler.morning_job import run_morning_job
        report_path = run_morning_job(
            game_date=game_date,
            send_email=not no_email,
            send_discord=not no_discord,
            dry_run=dry_run,
        )
        if report_path:
            print(f"\n✓ Report generated: {report_path}")
        else:
            print("\n✗ Pipeline completed but no report was generated.")
            sys.exit(1)
    except Exception as exc:
        logger.error("Morning pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


def cmd_update_elo(game_date: date, dry_run: bool) -> None:
    """Fetch results and update Elo ratings for a specific date."""
    logger.info("Updating Elo ratings from results on %s...", game_date.isoformat())
    try:
        from data.ingestors.mlb_results import batch_fetch_results
        from ensemble.elo_system import batch_update_from_results

        results = batch_fetch_results(game_date)
        if not results:
            logger.info("No results found for %s.", game_date.isoformat())
            return

        if dry_run:
            logger.info("DRY RUN: would update Elo for %d games.", len(results))
            for r in results:
                print(f"  {r.get('away_team_abbr','?')} @ {r.get('home_team_abbr','?')} "
                      f"— {'home' if r.get('home_win') else 'away'} won "
                      f"{r.get('home_score','?')}-{r.get('away_score','?')}")
        else:
            updates = batch_update_from_results(results, game_date.year)
            logger.info("Elo updated for %d games.", len(updates))
            for u in updates:
                print(f"  home: {u.get('home_elo_before',0):.0f} → {u.get('home_elo_after',0):.0f} "
                      f"| away: {u.get('away_elo_before',0):.0f} → {u.get('away_elo_after',0):.0f}")

    except Exception as exc:
        logger.error("Elo update failed: %s", exc, exc_info=True)
        sys.exit(1)


def cmd_backtest(backtest_date: date, dry_run: bool) -> None:
    """Run full pipeline on a historical date for validation."""
    logger.info("Running backtest pipeline for %s...", backtest_date.isoformat())
    try:
        from scheduler.morning_job import run_morning_job
        report_path = run_morning_job(
            game_date=backtest_date,
            send_email=False,
            send_discord=False,
            dry_run=dry_run,
        )
        if report_path:
            print(f"✓ Backtest report: {report_path}")
        else:
            print("✗ Backtest pipeline failed — check logs.")
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        sys.exit(1)


def cmd_scheduler(dry_run: bool) -> None:
    """Start the APScheduler background process."""
    try:
        from scheduler.main import start_scheduler
        start_scheduler(dry_run=dry_run)
    except Exception as exc:
        logger.error("Scheduler failed to start: %s", exc, exc_info=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_fetch_summary(rows: list, game_date: date) -> None:
    bar   = "=" * 72
    col_w = [10, 24, 8, 8, 8, 8, 8, 8]

    def _fmt(*fields) -> str:
        return "  ".join(str(f)[:w].ljust(w) for f, w in zip(fields, col_w))

    print(f"\n{bar}")
    print(f"  FETCH SUMMARY — {game_date.isoformat()}")
    print(f"{bar}")
    print(f"  {_fmt('Game PK','Matchup','H-BF','A-BF','H-Conf','A-Conf','H-Rost','A-Rost')}")
    print(f"  {'-' * (sum(col_w) + 2 * len(col_w))}")
    for row in rows:
        print(f"  {_fmt(row['game_pk'], row['matchup'], row['home_pitcher_bf'] or '—', row['away_pitcher_bf'] or '—', row['home_pitcher_conf'], row['away_pitcher_conf'], row['home_roster_size'] or '—', row['away_roster_size'] or '—')}")
        if row.get("weather_note") and row["weather_note"] != "—":
            print(f"       Weather: {row['weather_note']}")
        for err in row.get("errors", []):
            print(f"       [WARN] {err}")
    print(f"\n{bar}")
    print("  H/A = Home/Away  |  BF = Batters Faced YTD  |  Conf = data confidence")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # DB init (ensure tables exist)
    try:
        from data.cache.db import init_db
        init_db()
    except Exception as exc:
        logger.warning("DB init warning: %s", exc)

    game_date = _parse_date(args.date)

    # ── Route to command ──────────────────────────────────────────
    if args.scheduler:
        cmd_scheduler(dry_run=args.dry_run)

    elif args.backtest_date:
        backtest_date = _parse_date(args.backtest_date)
        cmd_backtest(backtest_date, dry_run=args.dry_run)

    elif args.update_elo:
        cmd_update_elo(game_date, dry_run=args.dry_run)

    elif args.simulate:
        cmd_simulate(args.game_pk, game_date)

    elif args.report_today or (
        not args.fetch_only and not args.report and
        not args.update_elo and not args.simulate and
        not args.scheduler and not args.backtest_date
    ):
        # Default: run full morning pipeline
        cmd_report_today(
            game_date=game_date,
            dry_run=args.dry_run,
            no_email=args.no_email,
            no_discord=args.no_discord,
        )

    elif args.fetch_only:
        cmd_fetch_only(game_date, args.game_pk)

    elif args.report:
        # Placeholder: generate report from cached DB results
        logger.warning(
            "--report from cached DB data is not yet fully implemented. "
            "Use --report-today to run the full pipeline."
        )
        cmd_report_today(
            game_date=game_date, dry_run=True,
            no_email=True, no_discord=True,
        )


if __name__ == "__main__":
    main()
