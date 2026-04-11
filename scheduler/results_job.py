"""
Apex Analytics — End-of-Day Results & Accuracy Job
Runs at 11:00 PM PT daily (after all West Coast games finish).

Purpose:
  - Fetch final scores for today's games via MLB Stats API
  - Compare against our morning/pregame predictions
  - Compute daily accuracy + Brier score
  - Update CalibrationHistory with actual outcomes (for season tracking)
  - Send a results email summarizing prediction vs. actual

Returns:
  dict with:
    n_final        : int  — games with final scores
    n_correct      : int  — predictions we got right (>50% → winner)
    accuracy       : float — daily hit rate (0-1)
    brier_score    : float — Brier score for the day (lower = better)
    email_sent     : bool
"""

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def run_results_job(
    game_date:    Optional[date] = None,
    send_email:   bool = True,
    send_discord: bool = False,
    dry_run:      bool = False,
) -> dict:
    """
    Execute the end-of-day results pipeline.

    Parameters
    ----------
    game_date    : Date to evaluate (defaults to today).
    send_email   : Send results email.
    send_discord : Post Discord results summary.
    dry_run      : Skip notifications and DB writes.

    Returns
    -------
    dict with n_final, n_correct, accuracy, brier_score, email_sent.
    """
    if game_date is None:
        game_date = date.today()

    logger.info("=" * 60)
    logger.info("APEX ANALYTICS — END-OF-DAY RESULTS — %s", game_date.isoformat())
    logger.info("=" * 60)

    result = {
        "n_final":    0,
        "n_correct":  0,
        "accuracy":   0.0,
        "brier_score": 0.0,
        "email_sent": False,
    }

    # ── 1. Fetch today's game PKs from DB ─────────────────────────
    try:
        from data.cache.db import get_session, Game, SimulationResult
        with get_session() as session:
            game_rows = session.query(Game).filter(
                Game.game_date == game_date.isoformat()
            ).all()
            game_map = {g.game_pk: g for g in game_rows}

        if not game_map:
            # Fallback: fetch schedule and build game_map from it
            from data.ingestors.mlb_schedule import fetch_schedule
            from types import SimpleNamespace
            schedule = fetch_schedule(game_date)
            game_map = {
                g["game_pk"]: SimpleNamespace(
                    game_pk        = g["game_pk"],
                    home_team_abbr = g.get("home_team_abbr", "HOME"),
                    away_team_abbr = g.get("away_team_abbr", "AWAY"),
                    venue_name     = g.get("venue_name", ""),
                )
                for g in schedule
            }

        game_pks = list(game_map.keys())

        logger.info("Checking results for %d games...", len(game_pks))
    except Exception as exc:
        logger.error("Could not load today's games: %s", exc)
        return result

    # ── 2. Fetch final scores from MLB API ────────────────────────
    from data.ingestors.mlb_results import fetch_game_result
    final_results = {}   # game_pk → {home_score, away_score, home_win, innings}

    for game_pk in game_pks:
        try:
            r = fetch_game_result(game_pk)
            if r and r.get("status") == "final":
                final_results[game_pk] = r
                logger.info(
                    "  %s @ %s — Final: %d–%d (%s wins)",
                    game_map[game_pk].away_team_abbr if game_pk in game_map else "???",
                    game_map[game_pk].home_team_abbr if game_pk in game_map else "???",
                    r["away_score"], r["home_score"],
                    (game_map[game_pk].home_team_abbr if r["home_win"] else game_map[game_pk].away_team_abbr)
                    if game_pk in game_map else ("HOME" if r["home_win"] else "AWAY"),
                )
        except Exception as exc:
            logger.debug("  game_pk=%d result fetch failed: %s", game_pk, exc)

    result["n_final"] = len(final_results)
    logger.info("%d of %d games are final.", len(final_results), len(game_pks))

    if not final_results:
        logger.info("No final results yet — results job will retry at next scheduled run.")
        return result

    # ── 3. Load our predictions from SimulationResult ─────────────
    # Prefer pregame prediction; fall back to morning.
    predictions = {}  # game_pk → calibrated_prob (home win %)
    try:
        with get_session() as session:
            sim_rows = session.query(SimulationResult).filter(
                SimulationResult.game_date == game_date.isoformat()
            ).all()

        # Build: prefer pregame over morning
        morning_preds = {}
        pregame_preds = {}
        for row in sim_rows:
            if row.report_type == "morning":
                morning_preds[row.game_pk] = row
            elif row.report_type == "pregame":
                pregame_preds[row.game_pk] = row

        for game_pk in final_results:
            row = pregame_preds.get(game_pk) or morning_preds.get(game_pk)
            if row:
                predictions[game_pk] = row.calibrated_prob or 0.5

        logger.info("Loaded predictions for %d/%d final games.", len(predictions), len(final_results))
    except Exception as exc:
        logger.error("Could not load predictions: %s", exc)

    # ── 4. Compute accuracy metrics ───────────────────────────────
    game_results = []  # list of dicts for email building

    for game_pk, final in final_results.items():
        game_info = game_map.get(game_pk)
        home_abbr = game_info.home_team_abbr if game_info else "HOME"
        away_abbr = game_info.away_team_abbr if game_info else "AWAY"
        venue     = game_info.venue_name if game_info else ""

        pred_prob = predictions.get(game_pk)   # home win probability (None if no prediction)
        home_won  = final["home_win"]
        home_score = final["home_score"]
        away_score = final["away_score"]
        innings    = final.get("innings", 9)

        if pred_prob is not None:
            predicted_home_win = pred_prob >= 0.50
            correct = (predicted_home_win == home_won)
        else:
            correct = None  # No prediction available

        game_results.append({
            "game_pk":    game_pk,
            "home_abbr":  home_abbr,
            "away_abbr":  away_abbr,
            "venue":      venue,
            "home_score": home_score,
            "away_score": away_score,
            "innings":    innings,
            "home_won":   home_won,
            "pred_prob":  pred_prob,
            "correct":    correct,
        })

    # Aggregate where we had predictions
    graded = [g for g in game_results if g["correct"] is not None]
    n_correct   = sum(1 for g in graded if g["correct"])
    n_graded    = len(graded)
    accuracy    = n_correct / n_graded if n_graded else 0.0
    brier_score = (
        sum((g["pred_prob"] - (1 if g["home_won"] else 0)) ** 2 for g in graded) / n_graded
        if n_graded else 0.0
    )

    result["n_correct"]   = n_correct
    result["n_final"]     = len(final_results)
    result["accuracy"]    = round(accuracy, 4)
    result["brier_score"] = round(brier_score, 4)

    logger.info(
        "Daily accuracy: %d/%d (%.1f%%) | Brier: %.4f",
        n_correct, n_graded, accuracy * 100, brier_score,
    )

    # ── 5. Update CalibrationHistory with actual outcomes ─────────
    if not dry_run:
        _update_calibration_outcomes(final_results, predictions, game_date)

    # ── 6. Load season stats for email ────────────────────────────
    season_stats = _compute_season_stats(game_date)

    # ── 7. Send results email ─────────────────────────────────────
    if not dry_run and send_email and game_results:
        try:
            from notification.email import send_results_email
            ok = send_results_email(
                game_results  = game_results,
                game_date     = game_date,
                daily_correct = n_correct,
                daily_total   = n_graded,
                daily_accuracy= accuracy,
                daily_brier   = brier_score,
                season_stats  = season_stats,
            )
            result["email_sent"] = ok
            if ok:
                logger.info("Results email sent.")
        except Exception as exc:
            logger.warning("Results email failed: %s", exc)

    logger.info(
        "END-OF-DAY RESULTS COMPLETE — %d final, %d/%d correct",
        len(final_results), n_correct, n_graded,
    )
    logger.info("=" * 60)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_calibration_outcomes(
    final_results: dict,
    predictions:   dict,
    game_date:     date,
) -> None:
    """Update CalibrationHistory rows with actual game outcomes."""
    try:
        from data.cache.db import get_session, CalibrationHistory

        with get_session() as session:
            for game_pk, final in final_results.items():
                row = session.query(CalibrationHistory).filter(
                    CalibrationHistory.game_pk == game_pk,
                ).first()

                actual = 1.0 if final["home_win"] else 0.0

                if row:
                    row.actual_outcome = actual
                else:
                    # Create record if morning job didn't (shouldn't happen but be safe)
                    pred = predictions.get(game_pk, 0.5)
                    session.add(CalibrationHistory(
                        game_pk        = game_pk,
                        game_date      = game_date.isoformat(),
                        ensemble_prob  = pred,
                        actual_outcome = actual,
                    ))

        logger.debug("CalibrationHistory updated for %d games.", len(final_results))
    except Exception as exc:
        logger.warning("CalibrationHistory update failed (non-critical): %s", exc)


def _compute_season_stats(game_date: date) -> dict:
    """
    Compute season-to-date accuracy from SimulationResult + Game tables
    for all games where actual outcome is known (home_win not null).
    """
    try:
        from data.cache.db import get_session, SimulationResult, Game
        season_start = f"{game_date.year}-01-01"

        with get_session() as session:
            # Join SimulationResult with Game where home_win is recorded
            sim_rows = session.query(SimulationResult, Game).join(
                Game, SimulationResult.game_pk == Game.game_pk
            ).filter(
                SimulationResult.game_date >= season_start,
                SimulationResult.game_date <= game_date.isoformat(),
                SimulationResult.report_type.in_(["morning", "pregame"]),
                Game.home_win.isnot(None),
            ).all()

        # Prefer pregame over morning for each game
        best_preds = {}
        for sim, game in sim_rows:
            existing = best_preds.get(sim.game_pk)
            if existing is None or sim.report_type == "pregame":
                best_preds[sim.game_pk] = (sim, game)

        records = list(best_preds.values())
        if not records:
            return {"n_games": 0, "n_correct": 0, "accuracy": 0.0, "brier_score": 0.0}

        probs   = [sim.calibrated_prob or 0.5 for sim, _ in records]
        actuals = [int(game.home_win) for _, game in records]
        n_correct   = sum(int((p >= 0.5) == bool(a)) for p, a in zip(probs, actuals))
        brier_score = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs)

        return {
            "n_games":    len(records),
            "n_correct":  n_correct,
            "accuracy":   round(n_correct / len(records), 4),
            "brier_score": round(brier_score, 4),
        }

    except Exception as exc:
        logger.debug("Season stats computation failed: %s", exc)
        return {"n_games": 0, "n_correct": 0, "accuracy": 0.0, "brier_score": 0.0}
