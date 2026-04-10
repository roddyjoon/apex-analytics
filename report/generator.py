"""
Apex Analytics — Report Generator
Full orchestration: game data + simulation results → rendered HTML report.

Pipeline:
  1. Accept list of game_data dicts (context, MC results, ensemble results)
  2. Generate charts for each game (run dist + win prob bars)
  3. Run explainer for key factors + weather strings
  4. Load rolling accuracy stats from DB
  5. Render Jinja2 templates → single self-contained HTML file
  6. Write to reports/YYYY-MM-DD/{report_type}/report.html
  7. Return absolute file path

Report types:
  "morning"  — 8:00 AM PT, projected lineups flagged
  "pregame"  — 1:00 PM PT, confirmed lineups where available
"""

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import jinja2

from config import MODELS_DIR
from report.charts   import run_distribution_chart, win_probability_chart
from report.explainer import (
    explain_game, weather_summary,
    lineup_confidence_summary, pitcher_confidence_label,
)

logger = logging.getLogger(__name__)

# Report output directory (relative to project root)
REPORT_BASE_DIR = Path(__file__).parent.parent / "reports"

# Jinja2 template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"

MODEL_VERSION = "1.0.0"


def generate_daily_report(
    game_date:   date,
    games_data:  list,
    report_type: str = "morning",
    postponements: Optional[list] = None,
) -> str:
    """
    Generate a complete HTML daily report for all games on a given date.

    Parameters
    ----------
    game_date    : Date of the games (used for file naming + display).
    games_data   : list of dicts, each containing:
                     {
                       "game_context":    GameContext,
                       "mc_result":       dict (from monte_carlo.run_monte_carlo),
                       "ensemble_result": dict (from blender.blend_predictions),
                       "game_time_et":    str  (e.g., "7:10 PM ET"),
                       "stadium_name":    str,
                       "home_full":       str  (full team name),
                       "away_full":       str,
                     }
    report_type  : "morning" or "pregame"
    postponements: list of team abbreviation pairs for postponed games

    Returns
    -------
    str — absolute path to the generated HTML file.
    """
    logger.info(
        "Generating %s report for %s (%d games)...",
        report_type, game_date.isoformat(), len(games_data),
    )

    # Load Jinja2 environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html"]),
        undefined=jinja2.Undefined,   # Silent for missing vars
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add custom filters
    env.filters["abs"] = abs

    # Build per-game context dicts
    game_contexts = []
    for gd in games_data:
        try:
            game_ctx = _build_game_template_context(gd, game_date)
            game_contexts.append(game_ctx)
        except Exception as exc:
            game_pk = gd.get("game_context", {})
            logger.error(
                "Failed to build template context for game: %s", exc, exc_info=True
            )
            continue

    # Load accuracy stats from DB
    accuracy_stats = _load_accuracy_stats(game_date)

    # Ensemble weights for today
    from ensemble.blender import get_phase_weights_for_date
    weights = get_phase_weights_for_date(game_date)

    # Build master template context
    template_context = {
        "report_date":       game_date.isoformat(),
        "report_date_long":  game_date.strftime("%A, %B %-d, %Y"),
        "report_type":       report_type,
        "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M PT"),
        "model_version":     MODEL_VERSION,
        "n_games":           len(game_contexts),
        "games":             game_contexts,
        "postponements":     postponements or [],
        "ensemble_phase":    weights.get("phase_label", "Mid-Season"),
        "ensemble_weights":  weights,

        # Accuracy stats
        "brier_30d":           accuracy_stats.get("brier_30d",  0.240),
        "accuracy_30d":        accuracy_stats.get("accuracy_30d", 0.540),
        "season_record":       accuracy_stats.get("season_record", "—"),
        "season_win_pct":      accuracy_stats.get("season_win_pct", None),
        "calibration_status":  accuracy_stats.get("calibration_status", "UNKNOWN"),
        "calibration_method":  accuracy_stats.get("calibration_method", "none"),
        "calibration_n_games": accuracy_stats.get("n_games", 0),
    }

    # Render templates
    try:
        template = env.get_template("daily_report.html")
        html     = template.render(**template_context)
    except Exception as exc:
        logger.error("Template rendering failed: %s", exc, exc_info=True)
        raise

    # Write to file
    output_path = _write_report(html, game_date, report_type)
    logger.info("Report written to %s", output_path)

    # Optional: generate PDF (requires WeasyPrint)
    _try_write_pdf(html, output_path)

    return str(output_path)


def _build_game_template_context(gd: dict, game_date: date) -> dict:
    """
    Build the full template context dict for one game card.
    Extracts everything from GameContext, mc_result, and ensemble_result.
    """
    ctx      = gd["game_context"]
    mc       = gd.get("mc_result", {})
    ensemble = gd.get("ensemble_result", {})

    home_sp = ctx.home_starter
    away_sp = ctx.away_starter
    park    = ctx.park

    # Charts
    run_dist      = mc.get("run_distribution", {5: 3, 25: 7, 50: 9, 75: 12, 95: 17})
    run_dist_b64  = run_distribution_chart(
        run_dist, ctx.home_team_abbr, ctx.away_team_abbr
    )
    # Use raw_ensemble as headline — it reflects actual model differentiation.
    # calibrated_prob is shown separately in the ensemble breakdown section.
    # Note: calibrated_prob is trained on 2024 backtest data and can flatten
    # early-season predictions to 50.5% when model outputs cluster near 50%.
    home_win_pct  = ensemble.get("raw_ensemble", ensemble.get("calibrated_prob", 0.53))
    away_win_pct  = 1.0 - home_win_pct
    winprob_b64   = win_probability_chart(
        home_win_pct, away_win_pct,
        ctx.home_team_abbr, ctx.away_team_abbr,
        ci_band=ensemble.get("confidence_band"),
    )

    # Key factors
    try:
        key_factors = explain_game(ctx, mc, ensemble)
    except Exception as exc:
        logger.warning("explain_game failed: %s", exc)
        key_factors = ["Factor analysis unavailable"]

    # Weather summary
    try:
        weather = weather_summary(park)
    except Exception as exc:
        logger.warning("weather_summary failed: %s", exc)
        weather = {"temp_str": "N/A", "wind_str": "N/A", "dome_flag": False,
                   "net_adj_str": "N/A", "conditions": "UNKNOWN"}

    # Lineup confidence
    home_conf = lineup_confidence_summary(ctx.home_lineup)
    away_conf = lineup_confidence_summary(ctx.away_lineup)

    # CI bounds
    mc_ci    = mc.get("confidence_interval", (0.43, 0.63))
    ens_band = ensemble.get("confidence_band", (0.43, 0.63))

    return {
        "game_pk":          ctx.game_pk,
        "away_abbr":        ctx.away_team_abbr,
        "home_abbr":        ctx.home_team_abbr,
        "away_full":        gd.get("away_full", ctx.away_team_abbr),
        "home_full":        gd.get("home_full", ctx.home_team_abbr),
        "game_time_et":     gd.get("game_time_et", "TBD"),
        "stadium_name":     gd.get("stadium_name", getattr(park, "venue_name", getattr(park, "stadium_name", "Stadium"))),
        "is_doubleheader":  gd.get("is_doubleheader", False),
        "doubleheader_game": gd.get("doubleheader_game", 1),

        # Win probability
        "home_win_pct":   round(home_win_pct, 4),
        "away_win_pct":   round(away_win_pct, 4),
        "ci_lower":       round(ens_band[0], 4),
        "ci_upper":       round(ens_band[1], 4),
        "ensemble_std":   round(abs(ens_band[1] - ens_band[0]) / 2, 4),

        # Projected score
        "projected_home_runs": mc.get("projected_home_runs", 4.5),
        "projected_away_runs": mc.get("projected_away_runs", 4.2),
        "projected_total":     mc.get("projected_total", 8.7),
        "extra_innings_pct":   mc.get("extra_innings_pct", 0.08),
        "avg_innings":         mc.get("avg_innings", 9.0),
        "run_distribution":    run_dist,

        # Charts
        "run_dist_chart_b64": run_dist_b64,
        "win_prob_chart_b64": winprob_b64,

        # Pitchers
        "home_starter":            home_sp,
        "away_starter":            away_sp,
        "home_starter_confidence": pitcher_confidence_label(home_sp),
        "away_starter_confidence": pitcher_confidence_label(away_sp),

        # Bullpens
        "home_bullpen": ctx.home_bullpen,
        "away_bullpen": ctx.away_bullpen,

        # Lineups
        "home_lineup":            ctx.home_lineup,
        "away_lineup":            ctx.away_lineup,
        "home_lineup_confidence": home_conf,
        "away_lineup_confidence": away_conf,

        # Park
        "park_factor_runs": getattr(park, "run_factor", 1.00),
        "park_factor_hr":   getattr(park, "hr_factor", 1.00),

        # Weather
        "weather": weather,

        # Key factors
        "key_factors": key_factors,

        # Ensemble breakdown
        "ensemble": ensemble,
    }


def _load_accuracy_stats(game_date: date) -> dict:
    """
    Load rolling accuracy + calibration stats from DB.
    Returns safe defaults if DB is unavailable or empty.
    """
    defaults = {
        "brier_30d":          0.240,
        "accuracy_30d":       0.540,
        "season_record":      "—",
        "season_win_pct":     None,
        "calibration_status": "UNKNOWN",
        "calibration_method": "none",
        "n_games":            0,
    }

    try:
        from ensemble.calibrator import get_calibrator, check_calibration_health

        cal    = get_calibrator()
        health = check_calibration_health(window_days=30)

        defaults["calibration_method"] = cal.method
        defaults["n_games"]            = cal.n_games
        defaults["brier_30d"]          = health.get("brier_calibrated", 0.240)
        defaults["accuracy_30d"]       = health.get("accuracy", 0.540)
        defaults["calibration_status"] = health.get("status", "UNKNOWN")

    except Exception as exc:
        logger.debug("Could not load accuracy stats from DB: %s", exc)

    try:
        from data.cache.db import get_session, AccuracyLog
        from datetime import timedelta

        season = game_date.year
        season_start = date(season, 3, 1)

        with get_session() as session:
            rows = session.query(AccuracyLog).filter(
                AccuracyLog.game_date >= season_start.isoformat()
            ).all()

        if rows:
            wins   = sum(1 for r in rows if getattr(r, "correct_prediction", False))
            total  = len(rows)
            pct    = wins / total if total > 0 else 0.0
            defaults["season_record"]  = f"{wins}-{total - wins}"
            defaults["season_win_pct"] = pct

    except Exception as exc:
        logger.debug("Could not load season record: %s", exc)

    return defaults


def _write_report(html: str, game_date: date, report_type: str) -> Path:
    """Write HTML report to disk. Returns the output file path."""
    out_dir  = REPORT_BASE_DIR / game_date.isoformat() / report_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.html"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


def _try_write_pdf(html: str, html_path: Path) -> None:
    """
    Optionally generate a PDF alongside the HTML.
    Requires WeasyPrint installed. Silently skips if unavailable.
    """
    try:
        import weasyprint
        pdf_path = html_path.with_suffix(".pdf")
        weasyprint.HTML(string=html).write_pdf(str(pdf_path))
        logger.info("PDF report written to %s", pdf_path)
    except ImportError:
        pass  # WeasyPrint not installed — skip silently
    except Exception as exc:
        logger.warning("PDF generation failed (non-critical): %s", exc)


def get_report_path(game_date: date, report_type: str) -> Path:
    """Return the expected output path for a report without generating it."""
    return REPORT_BASE_DIR / game_date.isoformat() / report_type / "report.html"


def list_existing_reports() -> list:
    """
    Return list of all existing report paths, newest first.
    Useful for the web viewer or archiving.
    """
    if not REPORT_BASE_DIR.exists():
        return []

    reports = []
    for date_dir in sorted(REPORT_BASE_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for type_dir in date_dir.iterdir():
            report_file = type_dir / "report.html"
            if report_file.exists():
                reports.append({
                    "date":        date_dir.name,
                    "report_type": type_dir.name,
                    "path":        str(report_file),
                    "size_kb":     round(report_file.stat().st_size / 1024, 1),
                })
    return reports
