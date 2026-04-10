"""
Apex Analytics — Morning Pipeline Job
Runs at 8:00 AM PT (11:00 AM ET) daily.

Pipeline:
  1. Fetch today's schedule
  2. For each game: fetch probable pitchers + projected lineups (historical fallback)
  3. Fetch weather for each stadium
  4. Build full profile objects (pitchers, lineups, bullpens, park context)
  5. Run 7,000-iteration Monte Carlo simulation per game
  6. Blend four-layer ensemble (MC + Elo + RF + LR + calibration)
  7. Generate HTML report (projected lineup flags shown)
  8. Update Elo ratings from yesterday's results
  9. Save simulation results to DB
  10. Send notifications (email + Discord) if configured

Design principles:
  - Each game processed independently — one game failure never blocks others
  - Every external call wrapped in try/except with logged error + fallback
  - Full pipeline time logged at end
  - Returns report path on success, None on total failure
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def run_morning_job(
    game_date:     Optional[date] = None,
    send_email:    bool = True,
    send_discord:  bool = True,
    dry_run:       bool = False,
) -> Optional[str]:
    """
    Execute the full morning pipeline.

    Parameters
    ----------
    game_date    : Date to run for (defaults to today).
    send_email   : Whether to send email report (requires RESEND_API_KEY).
    send_discord : Whether to post Discord summary (requires DISCORD_WEBHOOK_URL).
    dry_run      : If True, skip notifications and DB writes (testing mode).

    Returns
    -------
    str — absolute path to generated HTML report, or None if pipeline failed entirely.
    """
    if game_date is None:
        game_date = date.today()

    pipeline_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("APEX ANALYTICS — MORNING PIPELINE — %s", game_date.isoformat())
    logger.info("=" * 60)

    # ── Step 1: Fetch schedule ─────────────────────────────────────
    try:
        from data.ingestors.mlb_schedule import fetch_schedule
        games = fetch_schedule(game_date)
        games = [g for g in games if g.get("status") != "postponed"]
        logger.info("Schedule: %d games today (%d after postponement filter)",
                    len(games) + sum(1 for g in fetch_schedule(game_date) if g.get("status") == "postponed"),
                    len(games))
    except Exception as exc:
        logger.error("Schedule fetch failed: %s — aborting pipeline.", exc)
        return None

    if not games:
        logger.info("No games scheduled for %s — nothing to do.", game_date.isoformat())
        return None

    # ── Step 2: Update Elo from yesterday's results ────────────────
    _update_elo_from_yesterday(game_date, dry_run)

    # ── Step 3: Process each game ──────────────────────────────────
    games_data     = []
    season         = game_date.year

    for game in games:
        game_pk   = game["game_pk"]
        home_abbr = game.get("home_team_abbr", "HOME")
        away_abbr = game.get("away_team_abbr", "AWAY")

        logger.info("Processing: %s @ %s (game_pk=%d)", away_abbr, home_abbr, game_pk)
        game_start = time.perf_counter()

        try:
            game_data = _process_single_game(
                game, game_date, season, report_type="morning"
            )
            if game_data:
                games_data.append(game_data)
                elapsed = time.perf_counter() - game_start
                home_pct = game_data["ensemble_result"].get("raw_ensemble",
                           game_data["ensemble_result"].get("calibrated_prob", 0.5))
                logger.info(
                    "  Done in %.1fs | %s %.1f%% — %s %.1f%%",
                    elapsed, home_abbr, home_pct * 100,
                    away_abbr, (1 - home_pct) * 100,
                )
        except Exception as exc:
            logger.error("  FAILED (%s @ %s game_pk=%d): %s",
                         away_abbr, home_abbr, game_pk, exc, exc_info=True)

    if not games_data:
        logger.error("All games failed to process — no report generated.")
        return None

    # ── Step 4: Generate report ────────────────────────────────────
    report_path = None
    try:
        from report.generator import generate_daily_report
        report_path = generate_daily_report(
            game_date=game_date,
            games_data=games_data,
            report_type="morning",
        )
        logger.info("Report generated: %s", report_path)
    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)

    # ── Step 5: Save simulation results to DB ─────────────────────
    if not dry_run:
        _save_results_to_db(games_data, game_date, report_type="morning")

    # ── Step 6: Send notifications ─────────────────────────────────
    if not dry_run and report_path:
        if send_email:
            _send_email_notification(report_path, game_date, games_data)
        if send_discord:
            _send_discord_notification(games_data, report_path, game_date)

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info(
        "MORNING PIPELINE COMPLETE — %d games processed in %.1fs",
        len(games_data), total_elapsed,
    )
    logger.info("=" * 60)

    return report_path


def _process_single_game(
    game:        dict,
    game_date:   date,
    season:      int,
    report_type: str,
) -> Optional[dict]:
    """
    Build full context + run simulation + blend ensemble for one game.
    Returns a games_data dict ready for the report generator, or None on failure.
    Any unexpected exception is caught and logged so the pipeline continues.
    """
    # Validate that the game dict has the required primary key
    if "game_pk" not in game:
        logger.error("_process_single_game called with missing 'game_pk': %s", game)
        return None

    game_pk   = game["game_pk"]
    home_id   = game.get("home_team_id", 0)
    away_id   = game.get("away_team_id", 0)
    home_abbr = game.get("home_team_abbr", "HOME")
    away_abbr = game.get("away_team_abbr", "AWAY")

    # ── Probable pitchers ──────────────────────────────────────────
    try:
        from data.ingestors.mlb_lineups import fetch_probable_pitchers
        pitchers = fetch_probable_pitchers(game_pk)
    except Exception as exc:
        logger.warning("  Probable pitchers fetch failed: %s — using TBD", exc)
        pitchers = {}

    home_pitcher_id = pitchers.get("home", {}).get("player_id")
    away_pitcher_id = pitchers.get("away", {}).get("player_id")

    # ── Build pitcher profiles ─────────────────────────────────────
    try:
        from data.processors.profile_builder import build_pitcher_profile
        home_pitcher_name = pitchers.get("home", {}).get("player_name", "TBD")
        away_pitcher_name = pitchers.get("away", {}).get("player_name", "TBD")
        home_starter = build_pitcher_profile(
            player_id=home_pitcher_id,
            player_name=home_pitcher_name,
            throws="R",           # Default; profile builder overrides from DB
            team_id=home_id,
            is_home=True,
            season=season,
            game_date=game_date,
        ) if home_pitcher_id else None
        away_starter = build_pitcher_profile(
            player_id=away_pitcher_id,
            player_name=away_pitcher_name,
            throws="R",           # Default; profile builder overrides from DB
            team_id=away_id,
            is_home=False,
            season=season,
            game_date=game_date,
        ) if away_pitcher_id else None
    except Exception as exc:
        logger.warning("  Pitcher profile build failed: %s — using defaults", exc)
        home_starter = None
        away_starter = None

    # ── Lineups ────────────────────────────────────────────────────
    # build_lineup returns a dict {"lineup": [slot_dicts...], "is_confirmed": bool, ...}
    # build_lineup_profiles converts those slot dicts into BatterProfile objects.
    home_lineup_meta = {"lineup": [], "is_confirmed": False, "confidence_level": "ROSTER"}
    away_lineup_meta = {"lineup": [], "is_confirmed": False, "confidence_level": "ROSTER"}
    try:
        from data.processors.lineup_builder import build_lineup
        from data.processors.profile_builder import build_lineup_profiles
        home_lineup_meta = build_lineup(
            game_pk, home_id, home_abbr, game_date, report_type, season
        )
        away_lineup_meta = build_lineup(
            game_pk, away_id, away_abbr, game_date, report_type, season
        )
    except Exception as exc:
        logger.warning("  Lineup build failed: %s — using empty lineups", exc)

    # Convert slot dicts → BatterProfile objects
    try:
        from data.processors.profile_builder import build_lineup_profiles
        home_lineup = build_lineup_profiles(
            lineup_slots          = home_lineup_meta.get("lineup", []),
            opposing_pitcher_id   = pitchers.get("away", {}).get("player_id"),
            season                = season,
            game_date             = game_date,
            is_confirmed          = home_lineup_meta.get("is_confirmed", False),
        )
        away_lineup = build_lineup_profiles(
            lineup_slots          = away_lineup_meta.get("lineup", []),
            opposing_pitcher_id   = pitchers.get("home", {}).get("player_id"),
            season                = season,
            game_date             = game_date,
            is_confirmed          = away_lineup_meta.get("is_confirmed", False),
        )
    except Exception as exc:
        logger.warning("  Lineup profiles build failed: %s — using empty lineups", exc)
        home_lineup = []
        away_lineup = []

    # ── Bullpens ───────────────────────────────────────────────────
    try:
        from data.processors.bullpen_builder import build_bullpen_profile
        home_bullpen = build_bullpen_profile(
            home_id, home_abbr, game_date, is_home=True
        )
        away_bullpen = build_bullpen_profile(
            away_id, away_abbr, game_date, is_home=False
        )
    except Exception as exc:
        logger.warning("  Bullpen build failed: %s — using defaults", exc)
        from simulation.profiles import BullpenProfile
        home_bullpen = BullpenProfile(team_id=home_id, team_abbr=home_abbr)
        away_bullpen = BullpenProfile(team_id=away_id, team_abbr=away_abbr)

    # ── Park + weather ─────────────────────────────────────────────
    try:
        from data.processors.park_factors import get_park_factor_for_team
        from data.ingestors.mlb_schedule import get_venue_coords_by_team
        from data.ingestors.weather import fetch_weather
        from simulation.profiles import ParkContext

        pf     = get_park_factor_for_team(home_abbr)
        coords = get_venue_coords_by_team(home_abbr)
        is_dome    = coords.get("is_dome", False) if coords else False
        cf_deg     = coords.get("cf_orientation_deg", 0) if coords else 0
        elevation  = coords.get("elevation_ft", 0) if coords else 0

        # Weather (Open-Meteo)
        wind_run_adj = 0.0
        temp_run_adj = 0.0
        net_run_adj  = 0.0
        wind_speed   = 0.0
        wind_class   = "calm"
        weather_note = ""
        temp_f       = 72.0
        if coords and not is_dome:
            wx = fetch_weather(coords["lat"], coords["lon"], game_date)
            wind_speed   = wx.get("wind_speed_mph", 0.0)
            wind_class   = wx.get("wind_classification", "calm")
            wind_run_adj = wx.get("wind_run_adj", 0.0)
            temp_run_adj = wx.get("temp_run_adj", 0.0)
            net_run_adj  = wx.get("net_run_adj", 0.0)
            temp_f       = wx.get("temp_f", 72.0)
            weather_note = wx.get("weather_note", "")

        park = ParkContext(
            venue_id            = 0,
            venue_name          = coords.get("stadium_name", home_abbr) if coords else home_abbr,
            team_abbr           = home_abbr,
            run_factor          = pf.get("run_factor", 1.0),
            hr_factor           = pf.get("hr_factor", 1.0),
            temp_f              = temp_f,
            wind_speed_mph      = wind_speed,
            wind_classification = wind_class,
            wind_run_adj        = wind_run_adj,
            temp_run_adj        = temp_run_adj,
            net_run_adj         = net_run_adj,
            weather_note        = weather_note,
            is_dome             = is_dome,
            elevation_ft        = elevation,
        )
    except Exception as exc:
        logger.warning("  Park/weather build failed: %s — using neutral park", exc)
        from simulation.profiles import ParkContext
        park = ParkContext(venue_id=0, venue_name="Unknown Stadium",
                           team_abbr=home_abbr)

    # ── Elo ratings ────────────────────────────────────────────────
    try:
        from ensemble.elo_system import get_team_elo
        home_elo = get_team_elo(home_id, season)
        away_elo = get_team_elo(away_id, season)
    except Exception as exc:
        logger.warning("  Elo fetch failed: %s — using 1500", exc)
        home_elo = 1500.0
        away_elo = 1500.0

    # ── Recent form (decay win%) ────────────────────────────────────
    home_decay_wp, away_decay_wp = _fetch_decay_win_pct(
        home_id, away_id, game_date, season
    )

    # ── Assemble GameContext ───────────────────────────────────────
    from simulation.profiles import GameContext
    ctx = GameContext(
        game_pk=game_pk,
        game_date=game_date.isoformat(),
        home_team_id=home_id,
        home_team_abbr=home_abbr,
        away_team_id=away_id,
        away_team_abbr=away_abbr,
        home_starter=home_starter,
        away_starter=away_starter,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        home_bullpen=home_bullpen,
        away_bullpen=away_bullpen,
        park=park,
        home_elo=home_elo,
        away_elo=away_elo,
        home_decay_win_pct=home_decay_wp,
        away_decay_win_pct=away_decay_wp,
        report_type=report_type,
    )

    # ── Monte Carlo ────────────────────────────────────────────────
    try:
        from simulation.monte_carlo import run_monte_carlo
        from config import MONTE_CARLO_ITERATIONS
        mc_result = run_monte_carlo(ctx, n_iterations=MONTE_CARLO_ITERATIONS)
    except Exception as exc:
        logger.warning("  MC simulation failed: %s — using 50/50 baseline", exc)
        mc_result = _neutral_mc_result()

    # ── Ensemble blend ─────────────────────────────────────────────
    try:
        from ensemble.blender import blend_from_context
        ensemble_result = blend_from_context(mc_result, ctx, game_date)
    except Exception as exc:
        logger.warning("  Ensemble blend failed: %s — using MC probability", exc)
        mc_prob = mc_result.get("home_win_pct", 0.53)
        ensemble_result = {
            "mc_prob": mc_prob, "elo_prob": 0.53, "rf_prob": 0.53, "lr_prob": 0.53,
            "weights": {"mc": 1.0, "elo": 0.0, "rf": 0.0, "lr": 0.0},
            "phase_label": "fallback", "raw_ensemble": mc_prob,
            "calibrated_prob": mc_prob, "away_prob": 1.0 - mc_prob,
            "confidence_band": (max(0.05, mc_prob - 0.08), min(0.95, mc_prob + 0.08)),
        }

    return {
        "game_context":     ctx,
        "mc_result":        mc_result,
        "ensemble_result":  ensemble_result,
        "game_time_et":     game.get("game_time_et", "TBD"),
        "stadium_name":     game.get("venue_name", getattr(park, "venue_name", "Stadium")),
        "home_full":        game.get("home_team_name", home_abbr),
        "away_full":        game.get("away_team_name", away_abbr),
        "is_doubleheader":  game.get("double_header", "N") != "N",
        "doubleheader_game": 1 if game.get("double_header", "N") == "Y" else 2
                             if game.get("double_header", "N") == "Z" else 1,
    }


def _update_elo_from_yesterday(game_date: date, dry_run: bool) -> None:
    """Fetch yesterday's results and update Elo ratings."""
    yesterday = game_date - timedelta(days=1)
    try:
        from data.ingestors.mlb_results import batch_fetch_results
        from ensemble.elo_system import batch_update_from_results
        results = batch_fetch_results(yesterday)
        if results and not dry_run:
            updates = batch_update_from_results(results, yesterday.year)
            logger.info("Elo updated: %d games from %s", len(updates), yesterday.isoformat())
        elif results:
            logger.info("DRY RUN: would update Elo for %d games", len(results))
    except Exception as exc:
        logger.warning("Elo update from yesterday failed: %s", exc)


def _fetch_decay_win_pct(
    home_id: int, away_id: int, game_date: date, season: int
) -> tuple:
    """
    Fetch exponential-decay weighted win% for both teams (last 15 games).
    Returns (home_decay_wp, away_decay_wp) — defaults to 0.500 on failure.
    """
    try:
        from data.ingestors.mlb_results import fetch_team_recent_results
        from data.processors.recency_weighter import exponential_weighted_win_pct

        home_results = fetch_team_recent_results(home_id, game_date, n_games=15)
        away_results = fetch_team_recent_results(away_id, game_date, n_games=15)

        home_wp = exponential_weighted_win_pct(home_results, game_date) if home_results else 0.500
        away_wp = exponential_weighted_win_pct(away_results, game_date) if away_results else 0.500
        return home_wp, away_wp
    except Exception as exc:
        logger.debug("Decay win% fetch failed: %s — using 0.500", exc)
        return 0.500, 0.500


def _save_results_to_db(games_data: list, game_date: date, report_type: str) -> None:
    """Persist simulation results + ensemble outputs to DB calibration table."""
    try:
        from data.cache.db import get_session, SimulationResult, CalibrationHistory
        with get_session() as session:
            for gd in games_data:
                ctx      = gd["game_context"]
                mc       = gd["mc_result"]
                ensemble = gd["ensemble_result"]

                # Simulation result record
                sim_row = SimulationResult(
                    game_pk=ctx.game_pk,
                    game_date=game_date.isoformat(),
                    report_type=report_type,
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

                # Calibration history record (actual_outcome filled later by Elo updater)
                cal_row = CalibrationHistory(
                    game_pk=ctx.game_pk,
                    game_date=game_date.isoformat(),
                    ensemble_prob=ensemble.get("calibrated_prob", 0.53),
                    actual_outcome=None,  # Filled post-game
                )
                session.merge(cal_row)

        logger.info("Saved %d simulation results to DB.", len(games_data))
    except Exception as exc:
        logger.warning("DB save failed (non-critical): %s", exc)


def _send_email_notification(report_path: str, game_date: date, games_data: list = None) -> None:
    try:
        from notification.email import send_report_email
        ok = send_report_email(report_path, game_date, games_data=games_data)
        if ok:
            logger.info("Email report sent.")
        else:
            logger.info("Email skipped (RESEND_API_KEY not set or send failed).")
    except Exception as exc:
        logger.warning("Email notification failed: %s", exc)


def _send_discord_notification(
    games_data: list, report_path: str, game_date: date
) -> None:
    try:
        from notification.discord import post_discord_summary
        ok = post_discord_summary(games_data, report_path, game_date)
        if ok:
            logger.info("Discord summary posted.")
        else:
            logger.info("Discord skipped (webhook not configured or post failed).")
    except Exception as exc:
        logger.warning("Discord notification failed: %s", exc)


def _neutral_mc_result() -> dict:
    """Safe fallback MC result when simulation fails."""
    return {
        "home_win_pct": 0.530, "away_win_pct": 0.470,
        "projected_home_runs": 4.5, "projected_away_runs": 4.2,
        "projected_total": 8.7,
        "run_distribution": {5: 3, 10: 5, 25: 7, 50: 9, 75: 12, 90: 14, 95: 17},
        "home_run_distribution": {5: 1, 10: 2, 25: 3, 50: 5, 75: 7, 90: 9, 95: 11},
        "away_run_distribution": {5: 1, 10: 2, 25: 3, 50: 4, 75: 6, 90: 8, 95: 10},
        "confidence_interval": (0.460, 0.600),
        "extra_innings_pct": 0.080, "avg_innings": 9.0,
        "n_iterations": 0, "elapsed_seconds": 0.0, "iterations_per_sec": 0.0,
    }
