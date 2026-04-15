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
import requests
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def _fetch_final_scores(game_date: date) -> dict:
    """
    Fetch final scores for all games on a given date in a single API call.
    Uses the schedule endpoint with linescore hydration — more reliable than
    calling the linescore endpoint for each game individually.

    Returns dict of game_pk → {home_score, away_score, home_win, innings,
                                home_abbr, away_abbr, venue, is_final}
    """
    date_str = game_date.strftime("%m/%d/%Y")
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId":  1,
        "date":     date_str,
        "hydrate":  "linescore,team",
        "language": "en",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Schedule/scores fetch failed for %s: %s", game_date, exc)
        return {}

    games_out = {}
    dates = data.get("dates", [])
    if not dates:
        logger.info("No games found in MLB API for %s", game_date)
        return {}

    for game in dates[0].get("games", []):
        game_pk = game.get("gamePk")
        if not game_pk:
            continue

        status = game.get("status", {})
        abstract_state = status.get("abstractGameState", "")
        detailed_state = status.get("detailedState", "")
        is_final = abstract_state == "Final" or detailed_state == "Final"

        teams = game.get("teams", {})
        home_info  = teams.get("home", {})
        away_info  = teams.get("away", {})
        home_abbr  = home_info.get("team", {}).get("abbreviation", "HOME")
        away_abbr  = away_info.get("team", {}).get("abbreviation", "AWAY")
        venue      = game.get("venue", {}).get("name", "")

        linescore  = game.get("linescore", {})
        ls_teams   = linescore.get("teams", {})
        home_runs  = ls_teams.get("home", {}).get("runs")
        away_runs  = ls_teams.get("away", {}).get("runs")
        innings    = linescore.get("currentInning") or 9

        if is_final and home_runs is not None and away_runs is not None:
            games_out[game_pk] = {
                "home_score": home_runs,
                "away_score": away_runs,
                "home_win":   home_runs > away_runs,
                "innings":    innings,
                "home_abbr":  home_abbr,
                "away_abbr":  away_abbr,
                "venue":      venue,
                "is_final":   True,
            }
        else:
            logger.debug(
                "  game_pk=%d %s @ %s — state=%s/%s runs=%s/%s (not final yet)",
                game_pk, away_abbr, home_abbr, abstract_state, detailed_state,
                away_runs, home_runs,
            )

    return games_out


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
        "n_final":     0,
        "n_correct":   0,
        "accuracy":    0.0,
        "brier_score": 0.0,
        "email_sent":  False,
    }

    # ── 1. Fetch final scores (one API call, all games) ───────────
    logger.info("Fetching final scores from MLB API...")
    final_scores = _fetch_final_scores(game_date)

    result["n_final"] = len(final_scores)
    logger.info("%d final game(s) found for %s.", len(final_scores), game_date)

    if not final_scores:
        logger.info("No final results yet — will retry at next scheduled run.")
        return result

    for game_pk, g in sorted(final_scores.items()):
        logger.info(
            "  %s @ %s — Final: %d–%d (%s wins)",
            g["away_abbr"], g["home_abbr"],
            g["away_score"], g["home_score"],
            g["home_abbr"] if g["home_win"] else g["away_abbr"],
        )

    # ── 2. Load our predictions from SimulationResult ─────────────
    predictions = {}   # game_pk → calibrated_prob (home win %)
    try:
        from data.cache.db import get_session, SimulationResult
        with get_session() as session:
            sim_rows = session.query(SimulationResult).filter(
                SimulationResult.game_date == game_date.isoformat()
            ).all()

        morning_preds  = {}
        pregame_preds  = {}
        for row in sim_rows:
            if row.report_type == "morning":
                morning_preds[row.game_pk] = row
            elif row.report_type == "pregame":
                pregame_preds[row.game_pk] = row

        for game_pk in final_scores:
            row = pregame_preds.get(game_pk) or morning_preds.get(game_pk)
            if row and row.calibrated_prob is not None:
                predictions[game_pk] = row.calibrated_prob

        logger.info(
            "Predictions loaded: %d/%d games have a forecast.",
            len(predictions), len(final_scores),
        )
        if len(predictions) == 0:
            logger.warning(
                "No predictions found in DB for %s — "
                "SimulationResult table may be empty (check if morning job ran).",
                game_date,
            )
    except Exception as exc:
        logger.error("Could not load predictions from DB: %s", exc, exc_info=True)

    # ── 3. Build game_results list ────────────────────────────────
    game_results = []
    for game_pk, final in final_scores.items():
        pred_prob  = predictions.get(game_pk)
        home_won   = final["home_win"]
        correct    = None
        if pred_prob is not None:
            correct = (pred_prob >= 0.50) == home_won

        game_results.append({
            "game_pk":    game_pk,
            "home_abbr":  final["home_abbr"],
            "away_abbr":  final["away_abbr"],
            "venue":      final["venue"],
            "home_score": final["home_score"],
            "away_score": final["away_score"],
            "innings":    final["innings"],
            "home_won":   home_won,
            "pred_prob":  pred_prob,
            "correct":    correct,
        })

    # ── 4. Compute accuracy metrics ───────────────────────────────
    graded      = [g for g in game_results if g["correct"] is not None]
    n_correct   = sum(1 for g in graded if g["correct"])
    n_graded    = len(graded)
    accuracy    = n_correct / n_graded if n_graded else 0.0
    brier_score = (
        sum((g["pred_prob"] - (1 if g["home_won"] else 0)) ** 2 for g in graded) / n_graded
        if n_graded else 0.0
    )

    result["n_correct"]   = n_correct
    result["accuracy"]    = round(accuracy, 4)
    result["brier_score"] = round(brier_score, 4)

    logger.info(
        "Daily accuracy: %d/%d (%.1f%%) | Brier: %.4f",
        n_correct, n_graded, accuracy * 100, brier_score,
    )

    # ── 5. Persist actual outcomes to DB ──────────────────────────
    if not dry_run:
        _update_calibration_outcomes(final_scores, predictions, game_date)
        _persist_game_results(final_scores, game_date)

    # ── 6. Season stats for email ─────────────────────────────────
    season_stats = _compute_season_stats(game_date)

    # ── 7. Send email ─────────────────────────────────────────────
    if not dry_run and send_email:
        try:
            from notification.email import send_results_email
            ok = send_results_email(
                game_results   = game_results,
                game_date      = game_date,
                daily_correct  = n_correct,
                daily_total    = n_graded,
                daily_accuracy = accuracy,
                daily_brier    = brier_score,
                season_stats   = season_stats,
            )
            result["email_sent"] = ok
            logger.info("Results email %s.", "sent ✓" if ok else "FAILED")
        except Exception as exc:
            logger.error("Results email error: %s", exc, exc_info=True)

    logger.info(
        "END-OF-DAY RESULTS COMPLETE — %d final, %d/%d correct",
        len(final_scores), n_correct, n_graded,
    )
    logger.info("=" * 60)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persist_game_results(final_scores: dict, game_date: date) -> None:
    """Write final scores back to the Game table so season stats work."""
    try:
        from data.cache.db import get_session, Game
        with get_session() as session:
            for game_pk, g in final_scores.items():
                row = session.query(Game).filter(Game.game_pk == game_pk).first()
                if row:
                    row.home_score = g["home_score"]
                    row.away_score = g["away_score"]
                    row.innings    = g["innings"]
                    row.home_win   = g["home_win"]
                    row.status     = "final"
                else:
                    session.add(Game(
                        game_pk        = game_pk,
                        game_date      = game_date.isoformat(),
                        home_team_abbr = g["home_abbr"],
                        away_team_abbr = g["away_abbr"],
                        venue_name     = g["venue"],
                        home_score     = g["home_score"],
                        away_score     = g["away_score"],
                        home_win       = g["home_win"],
                        status         = "final",
                    ))
        logger.debug("Persisted %d game results to DB.", len(final_scores))
    except Exception as exc:
        logger.warning("Could not persist game results (non-critical): %s", exc)


def _update_calibration_outcomes(
    final_scores: dict,
    predictions:  dict,
    game_date:    date,
) -> None:
    """Update CalibrationHistory rows with actual game outcomes."""
    try:
        from data.cache.db import get_session, CalibrationHistory
        with get_session() as session:
            for game_pk, g in final_scores.items():
                actual = 1.0 if g["home_win"] else 0.0
                row = session.query(CalibrationHistory).filter(
                    CalibrationHistory.game_pk == game_pk,
                ).first()
                if row:
                    row.actual_outcome = actual
                else:
                    pred = predictions.get(game_pk, 0.5)
                    session.add(CalibrationHistory(
                        game_pk        = game_pk,
                        game_date      = game_date.isoformat(),
                        ensemble_prob  = pred,
                        actual_outcome = actual,
                    ))
        logger.debug("CalibrationHistory updated for %d games.", len(final_scores))
    except Exception as exc:
        logger.warning("CalibrationHistory update failed (non-critical): %s", exc)


def _compute_season_stats(game_date: date) -> dict:
    """
    Season-to-date accuracy from SimulationResult + Game tables
    (only games where actual outcome is recorded).
    """
    try:
        from data.cache.db import get_session, SimulationResult, Game
        season_start = f"{game_date.year}-01-01"

        with get_session() as session:
            sim_rows = session.query(SimulationResult, Game).join(
                Game, SimulationResult.game_pk == Game.game_pk
            ).filter(
                SimulationResult.game_date >= season_start,
                SimulationResult.game_date <= game_date.isoformat(),
                SimulationResult.report_type.in_(["morning", "pregame"]),
                Game.home_win.isnot(None),
            ).all()

        best_preds = {}
        for sim, game in sim_rows:
            existing = best_preds.get(sim.game_pk)
            if existing is None or sim.report_type == "pregame":
                best_preds[sim.game_pk] = (sim, game)

        records = list(best_preds.values())
        if not records:
            return {"n_games": 0, "n_correct": 0, "accuracy": 0.0, "brier_score": 0.0}

        probs   = [sim.calibrated_prob or 0.5 for sim, _ in records]
        actuals = [int(game.home_win)          for _, game in records]
        n_correct   = sum(int((p >= 0.5) == bool(a)) for p, a in zip(probs, actuals))
        brier_score = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs)

        return {
            "n_games":     len(records),
            "n_correct":   n_correct,
            "accuracy":    round(n_correct / len(records), 4),
            "brier_score": round(brier_score, 4),
        }

    except Exception as exc:
        logger.debug("Season stats computation failed: %s", exc)
        return {"n_games": 0, "n_correct": 0, "accuracy": 0.0, "brier_score": 0.0}
