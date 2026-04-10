"""
Apex Analytics — Discord Webhook Notifications
Posts a summary embed to a Discord channel after each report generation.

Requires:
  DISCORD_WEBHOOK_URL — from Discord server settings (Server Settings → Integrations → Webhooks)

Message format:
  One embed per report with:
  - Title: "⚾ Apex Analytics — April 9, 2025"
  - One line per game: "NYY @ LAD | 7:10 PM ET | NYY 41.0% — LAD 59.0%"
  - Footer: Brier score + model version

Silently skips if DISCORD_WEBHOOK_URL is not set.
"""

import logging
import os
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DISCORD_COLOR_BLUE  = 0x4A9EFF   # Home team color
DISCORD_COLOR_GREEN = 0x4CAF76   # Good/success
DISCORD_COLOR_GOLD  = 0xFFD700   # Report header


def post_discord_summary(
    games_data:   list,
    report_path:  str,
    game_date:    date,
    title:        Optional[str] = None,
    report_url:   Optional[str] = None,
) -> bool:
    """
    Post a summary embed to Discord with win probabilities for all games.

    Parameters
    ----------
    games_data  : List of game data dicts (from pipeline output).
    report_path : Path to the generated HTML report (used for file size display).
    game_date   : Date of the report.
    title       : Override the embed title.
    report_url  : URL to the hosted report (if deployed to Cloudflare Pages, etc.).

    Returns
    -------
    bool — True on success, False if skipped or failed.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        logger.debug("DISCORD_WEBHOOK_URL not set — Discord notification skipped.")
        return False

    if not games_data:
        logger.debug("No game data to post — Discord skipped.")
        return False

    # Build embed
    embed = _build_embed(games_data, game_date, title, report_url)

    try:
        response = requests.post(
            webhook_url,
            json={"embeds": [embed]},
            timeout=10,
        )
        if response.status_code in (200, 204):
            logger.info("Discord summary posted for %d games.", len(games_data))
            return True
        else:
            logger.warning(
                "Discord webhook returned %d: %s",
                response.status_code, response.text[:200],
            )
            return False

    except Exception as exc:
        logger.error("Discord post failed: %s", exc)
        return False


def post_failure_alert(job_name: str, error_msg: str) -> bool:
    """
    Post a pipeline failure alert to Discord.
    Used by the scheduler when a job crashes.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False

    embed = {
        "title":       "⚠️ Apex Analytics — Pipeline Failure",
        "description": f"**Job:** `{job_name}`\n**Error:** `{error_msg[:500]}`",
        "color":       0xFF4444,
        "footer":      {"text": "Check server logs for full stack trace."},
    }

    try:
        response = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        return response.status_code in (200, 204)
    except Exception as exc:
        logger.warning("Could not post Discord failure alert: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_embed(
    games_data:  list,
    game_date:   date,
    title:       Optional[str],
    report_url:  Optional[str],
) -> dict:
    """Build the Discord embed dict."""
    date_str = game_date.strftime("%A, %B %-d, %Y")

    if title is None:
        title = f"⚾ Apex Analytics — {date_str}"

    # Build game lines
    game_lines = []
    for gd in games_data:
        ctx      = gd.get("game_context")
        ensemble = gd.get("ensemble_result", {})
        if not ctx:
            continue

        home_abbr = getattr(ctx, "home_team_abbr", "HOME")
        away_abbr = getattr(ctx, "away_team_abbr", "AWAY")
        home_pct  = ensemble.get("calibrated_prob", 0.53) * 100
        away_pct  = (1 - ensemble.get("calibrated_prob", 0.53)) * 100
        game_time = gd.get("game_time_et", "TBD")

        # Visual lean indicator
        if home_pct >= 60:
            lean = f"🔵 **{home_abbr}**"
        elif away_pct >= 60:
            lean = f"🔴 **{away_abbr}**"
        else:
            lean = "⚪ Even"

        line = (
            f"`{away_abbr} @ {home_abbr}` · {game_time}\n"
            f"{away_abbr} {away_pct:.1f}% — {home_abbr} {home_pct:.1f}% · {lean}"
        )
        game_lines.append(line)

    description = "\n\n".join(game_lines) if game_lines else "No games today."

    # Footer with accuracy stats
    try:
        from ensemble.calibrator import get_calibrator
        cal      = get_calibrator()
        footer   = f"Calibration: {cal.method} · n={cal.n_games} games · Apex Analytics v1.0"
    except Exception:
        footer = "Apex Analytics — Educational MLB Win Probability Engine"

    embed = {
        "title":       title,
        "description": description,
        "color":       DISCORD_COLOR_GOLD,
        "footer":      {"text": footer},
        "timestamp":   date.today().isoformat() + "T12:00:00.000Z",
        "fields":      [],
    }

    # Add report link field if URL provided
    if report_url:
        embed["fields"].append({
            "name":   "📊 Full Report",
            "value":  f"[View complete analysis]({report_url})",
            "inline": False,
        })

    # Add game count field
    embed["fields"].append({
        "name":   "Games",
        "value":  str(len(games_data)),
        "inline": True,
    })

    return embed


def send_test_discord(message: str = "Apex Analytics — test message ✓") -> bool:
    """
    Send a test message to verify Discord webhook configuration.

    Usage:
      python -c "from notification.discord import send_test_discord; send_test_discord()"
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("ERROR: DISCORD_WEBHOOK_URL not set in environment.")
        return False

    try:
        response = requests.post(
            webhook_url,
            json={"content": message},
            timeout=10,
        )
        if response.status_code in (200, 204):
            print(f"✓ Discord test message sent.")
            return True
        else:
            print(f"✗ Discord returned {response.status_code}: {response.text[:200]}")
            return False
    except Exception as exc:
        print(f"✗ Discord test failed: {exc}")
        return False
