"""
Apex Analytics — Email Notification via Resend API
Sends a clean, Gmail-safe digest email with all games for the day.

Requires:
  RESEND_API_KEY    — from resend.com (free: 3,000 emails/month)
  REPORT_EMAIL_TO   — recipient email address
  REPORT_EMAIL_FROM — sender address (must be verified domain in Resend)

Silently skips if RESEND_API_KEY is not set (not a fatal error).
"""

import logging
import os
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


# ── Public send functions ──────────────────────────────────────────────────────

def send_report_email(
    report_html_path: str,
    game_date:        date,
    games_data:       Optional[list] = None,
    subject_suffix:   str = "",
    recipients:       Optional[list] = None,
) -> bool:
    """
    Send the daily report email via Resend.
    Builds a clean, Gmail-safe digest from games_data (preferred).
    Falls back to the raw report HTML if games_data not provided.
    """
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        logger.debug("RESEND_API_KEY not set — email notification skipped.")
        return False

    if recipients is None:
        to_env = os.environ.get("REPORT_EMAIL_TO", "").strip()
        if not to_env:
            logger.warning("REPORT_EMAIL_TO not set — email skipped.")
            return False
        recipients = [addr.strip() for addr in to_env.split(",") if addr.strip()]

    from_addr = os.environ.get("REPORT_EMAIL_FROM", "onboarding@resend.dev").strip()

    # Build email HTML
    if games_data:
        html_content = _build_digest_email(games_data, game_date, subject_suffix)
        n_games = len(games_data)
    else:
        # Fallback: read the full report file (may be large)
        try:
            from pathlib import Path
            html_content = Path(report_html_path).read_text(encoding="utf-8")
            n_games = html_content.count('class="game-card"')
        except Exception as exc:
            logger.error("Could not read report file %s: %s", report_html_path, exc)
            return False

    # Subject line
    date_str = game_date.strftime("%B %-d, %Y")
    report_label = subject_suffix or ("Morning Report" if not subject_suffix else subject_suffix)
    subject = f"⚾ Apex Analytics — {date_str} · {n_games} Games"
    if subject_suffix:
        subject += f" · {subject_suffix}"

    return _send_via_resend(api_key, from_addr, recipients, subject, html_content)


def send_test_email(recipient: str) -> bool:
    """Send a simple test email to verify Resend configuration."""
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RESEND_API_KEY not set in environment.")
        return False
    from_addr = os.environ.get("REPORT_EMAIL_FROM", "onboarding@resend.dev")
    html = (
        "<div style='font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:24px'>"
        "<h2 style='color:#1a56db'>⚾ Apex Analytics</h2>"
        "<p>Test email sent successfully. Your Resend configuration is working.</p>"
        "</div>"
    )
    ok = _send_via_resend(api_key, from_addr, [recipient],
                          "Apex Analytics — Test Email", html)
    print("✓ Test email sent" if ok else "✗ Test email failed")
    return ok


# ── Email builder ──────────────────────────────────────────────────────────────

def _build_digest_email(games_data: list, game_date: date, report_type: str = "") -> str:
    """
    Build a clean, Gmail-safe HTML email from games_data.
    Uses only inline CSS and table layouts — no CSS variables, no grid/flex.
    Target size: < 80 KB for 15 games.
    """
    date_long   = game_date.strftime("%A, %B %-d, %Y")
    n_games     = len(games_data)
    label       = "Pre-Game Update" if "pregame" in report_type.lower() else "Morning Report"

    games_html = "\n".join(_game_block(gd) for gd in games_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Apex Analytics — {date_long}</title>
</head>
<body style="margin:0;padding:0;background:#f4f6f9;font-family:Arial,Helvetica,sans-serif;">

<!-- Wrapper -->
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f6f9;padding:20px 0;">
<tr><td align="center">
<table width="620" cellpadding="0" cellspacing="0" style="max-width:620px;width:100%;">

  <!-- Header -->
  <tr>
    <td style="background:#1a56db;border-radius:8px 8px 0 0;padding:20px 24px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td>
            <div style="font-size:22px;font-weight:700;color:#ffffff;letter-spacing:1px;">
              ⚾ APEX ANALYTICS
            </div>
            <div style="font-size:13px;color:#bfdbfe;margin-top:3px;">{label}</div>
          </td>
          <td align="right">
            <div style="font-size:13px;color:#bfdbfe;text-align:right;">{date_long}</div>
            <div style="font-size:20px;font-weight:700;color:#ffffff;text-align:right;margin-top:4px;">
              {n_games} Games
            </div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Games -->
  {games_html}

  <!-- Footer -->
  <tr>
    <td style="background:#1e293b;border-radius:0 0 8px 8px;padding:16px 24px;text-align:center;">
      <p style="margin:0;font-size:11px;color:#64748b;line-height:1.6;">
        Apex Analytics · 7,000-iteration Monte Carlo + Elo + RF + LR ensemble<br>
        Data: Baseball Savant · MLB Stats API · Open-Meteo<br>
        <strong style="color:#475569;">Educational tool only. Not financial advice.</strong>
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def _game_block(gd: dict) -> str:
    """Render one game as an email-safe HTML table block."""
    ctx      = gd.get("game_context")
    mc       = gd.get("mc_result", {})
    ensemble = gd.get("ensemble_result", {})

    away_abbr  = gd.get("away_full", getattr(ctx, "away_team_abbr", "AWAY")) if ctx else gd.get("away_full", "AWAY")
    home_abbr  = gd.get("home_full", getattr(ctx, "home_team_abbr", "HOME")) if ctx else gd.get("home_full", "HOME")
    away_short = getattr(ctx, "away_team_abbr", away_abbr[:3].upper()) if ctx else away_abbr[:3].upper()
    home_short = getattr(ctx, "home_team_abbr", home_abbr[:3].upper()) if ctx else home_abbr[:3].upper()

    game_time  = gd.get("game_time_et", "TBD")
    stadium    = gd.get("stadium_name", "")

    home_pct   = ensemble.get("raw_ensemble", ensemble.get("calibrated_prob", 0.53))
    away_pct   = 1.0 - home_pct
    home_pct_i = int(round(home_pct * 100))
    away_pct_i = 100 - home_pct_i

    # Color: stronger side gets blue, weaker gets gray
    home_is_fav = home_pct >= 0.50
    home_pct_color = "#1a56db" if home_is_fav else "#64748b"
    away_pct_color = "#1a56db" if not home_is_fav else "#64748b"
    home_bar_color = "#1a56db" if home_is_fav else "#94a3b8"
    away_bar_color = "#1a56db" if not home_is_fav else "#94a3b8"

    # Bar widths (min 8% so text is visible)
    home_bar_w = max(8, home_pct_i)
    away_bar_w = max(8, away_pct_i)

    # Pitchers
    home_pitcher = "TBD"
    away_pitcher = "TBD"
    if ctx:
        if ctx.home_starter:
            hp = ctx.home_starter
            home_pitcher = getattr(hp, "player_name", "TBD")
            hand = getattr(hp, "throws", "")
            era  = getattr(hp, "era", None)
            if hand or era is not None:
                home_pitcher += f" ({hand}{'  ERA ' + f'{era:.2f}' if era else ''})"
        if ctx.away_starter:
            ap = ctx.away_starter
            away_pitcher = getattr(ap, "player_name", "TBD")
            hand = getattr(ap, "throws", "")
            era  = getattr(ap, "era", None)
            if hand or era is not None:
                away_pitcher += f" ({hand}{'  ERA ' + f'{era:.2f}' if era else ''})"

    # Projected score
    home_runs = mc.get("projected_home_runs", 0)
    away_runs = mc.get("projected_away_runs", 0)
    total     = mc.get("projected_total", 0)
    score_str = f"{away_runs:.1f} – {home_runs:.1f}  (O/U {total:.1f})"

    # Weather
    park = getattr(ctx, "park", None) if ctx else None
    weather_str = ""
    if park:
        if getattr(park, "is_dome", False):
            weather_str = "🏟 Dome — weather neutral"
        else:
            temp   = getattr(park, "temp_f", None)
            wind   = getattr(park, "wind_speed_mph", None)
            w_cls  = getattr(park, "wind_classification", "")
            note   = getattr(park, "weather_note", "")
            parts  = []
            if temp:  parts.append(f"{temp:.0f}°F")
            if wind:
                arrow = {"out": "💨 Out", "in": "💨 In", "calm": "🌤 Calm"}.get(w_cls, f"Wind {w_cls}")
                parts.append(f"{arrow} {wind:.0f} mph")
            if note:  parts.append(note)
            weather_str = "  ·  ".join(parts) if parts else "Weather data unavailable"

    # Confidence interval
    ci    = ensemble.get("confidence_band", (home_pct - 0.07, home_pct + 0.07))
    ci_lo = int(round(ci[0] * 100))
    ci_hi = int(round(ci[1] * 100))

    return f"""
  <!-- ── Game: {away_short} @ {home_short} ── -->
  <tr>
    <td style="padding:8px 0 0 0;">
      <table width="100%" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;">

        <!-- Game header bar -->
        <tr>
          <td style="background:#1e293b;padding:12px 20px;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td>
                  <span style="font-size:20px;font-weight:700;color:#f8fafc;">
                    {away_short}
                    <span style="color:#475569;font-weight:400;margin:0 6px;">@</span>
                    {home_short}
                  </span>
                  <div style="font-size:11px;color:#94a3b8;margin-top:2px;">
                    {away_abbr} @ {home_abbr}
                  </div>
                </td>
                <td align="right">
                  <div style="font-size:15px;font-weight:700;color:#f8fafc;">{game_time} ET</div>
                  <div style="font-size:11px;color:#94a3b8;">{stadium}</div>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Win Probability -->
        <tr>
          <td style="padding:16px 20px 12px 20px;border-bottom:1px solid #f1f5f9;">
            <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:10px;">Win Probability</div>

            <!-- Away team row -->
            <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:6px;">
              <tr>
                <td width="50" style="font-size:13px;font-weight:700;color:#1e293b;">{away_short}</td>
                <td style="padding:0 10px;">
                  <div style="background:#f1f5f9;border-radius:4px;height:24px;overflow:hidden;">
                    <div style="background:{away_bar_color};width:{away_bar_w}%;height:24px;
                                border-radius:4px;min-width:24px;"></div>
                  </div>
                </td>
                <td width="44" align="right"
                    style="font-size:22px;font-weight:700;color:{away_pct_color};">{away_pct_i}%</td>
              </tr>
            </table>

            <!-- Home team row -->
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="50" style="font-size:13px;font-weight:700;color:#1e293b;">{home_short}</td>
                <td style="padding:0 10px;">
                  <div style="background:#f1f5f9;border-radius:4px;height:24px;overflow:hidden;">
                    <div style="background:{home_bar_color};width:{home_bar_w}%;height:24px;
                                border-radius:4px;min-width:24px;"></div>
                  </div>
                </td>
                <td width="44" align="right"
                    style="font-size:22px;font-weight:700;color:{home_pct_color};">{home_pct_i}%</td>
              </tr>
            </table>

            <div style="font-size:10px;color:#94a3b8;margin-top:6px;">
              Confidence interval: {ci_lo}% – {ci_hi}%
            </div>
          </td>
        </tr>

        <!-- Pitchers + Score -->
        <tr>
          <td style="padding:12px 20px;border-bottom:1px solid #f1f5f9;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <!-- Pitchers -->
                <td width="60%" style="vertical-align:top;">
                  <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:6px;">Starting Pitchers</div>
                  <div style="font-size:12px;color:#334155;margin-bottom:4px;">
                    <strong style="color:#64748b;">{away_short}</strong>&nbsp; {away_pitcher}
                  </div>
                  <div style="font-size:12px;color:#334155;">
                    <strong style="color:#64748b;">{home_short}</strong>&nbsp; {home_pitcher}
                  </div>
                </td>
                <!-- Projected score -->
                <td width="40%" style="vertical-align:top;padding-left:16px;border-left:1px solid #f1f5f9;">
                  <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:6px;">Projected Score</div>
                  <div style="font-size:18px;font-weight:700;color:#1e293b;">{score_str}</div>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Weather -->
        {"" if not weather_str else f'''<tr>
          <td style="padding:10px 20px;background:#f8fafc;">
            <span style="font-size:11px;color:#64748b;">{weather_str}</span>
          </td>
        </tr>'''}

        <!-- Lineups -->
        {_lineup_section(ctx, away_short, home_short)}

      </table>
    </td>
  </tr>"""


def _lineup_section(ctx, away_short: str, home_short: str) -> str:
    """
    Render a two-column batting order table for both teams.
    Shows: order, name, position, xwOBA, OBP, SLG.
    Returns empty string if no lineup data available.
    """
    if ctx is None:
        return ""

    away_lineup = getattr(ctx, "away_lineup", []) or []
    home_lineup = getattr(ctx, "home_lineup", []) or []

    if not away_lineup and not home_lineup:
        return ""

    def _team_rows(lineup, is_confirmed: bool) -> str:
        if not lineup:
            return '<tr><td colspan="5" style="padding:6px;font-size:11px;color:#94a3b8;font-style:italic;">Lineup TBD</td></tr>'
        rows = []
        confirmed_label = (
            '<span style="font-size:9px;color:#16a34a;font-weight:700;'
            'background:#dcfce7;padding:1px 5px;border-radius:3px;margin-left:4px;">✓ CONFIRMED</span>'
            if is_confirmed else
            '<span style="font-size:9px;color:#d97706;font-weight:700;'
            'background:#fef3c7;padding:1px 5px;border-radius:3px;margin-left:4px;">PROJECTED</span>'
        )
        rows.append(
            f'<tr><td colspan="5" style="padding:4px 6px 6px 6px;">'
            f'<span style="font-size:10px;font-weight:700;color:#64748b;'
            f'text-transform:uppercase;letter-spacing:.5px;">Batting Order</span>'
            f'{confirmed_label}</td></tr>'
        )
        # Header row
        rows.append(
            '<tr style="background:#f8fafc;">'
            '<td style="padding:3px 6px;font-size:9px;color:#94a3b8;font-weight:700;width:18px;">#</td>'
            '<td style="padding:3px 6px;font-size:9px;color:#94a3b8;font-weight:700;">PLAYER</td>'
            '<td style="padding:3px 6px;font-size:9px;color:#94a3b8;font-weight:700;text-align:center;width:28px;">POS</td>'
            '<td style="padding:3px 6px;font-size:9px;color:#94a3b8;font-weight:700;text-align:right;width:40px;">xwOBA</td>'
            '<td style="padding:3px 6px;font-size:9px;color:#94a3b8;font-weight:700;text-align:right;width:40px;">OPS</td>'
            '</tr>'
        )
        for i, batter in enumerate(sorted(lineup, key=lambda b: getattr(b, "batting_order", 9))):
            name   = getattr(batter, "player_name", "")
            order  = getattr(batter, "batting_order", i + 1)
            pos    = getattr(batter, "position", "")
            xwoba  = getattr(batter, "xwoba", None)
            obp    = getattr(batter, "obp", None)
            slg    = getattr(batter, "slg", None)
            ops    = (obp + slg) if (obp and slg) else None

            # Color-code xwOBA: great ≥.380, good ≥.340, avg ≥.300
            if xwoba is not None:
                if xwoba >= 0.380:
                    xwoba_color = "#15803d"   # green
                elif xwoba >= 0.340:
                    xwoba_color = "#1a56db"   # blue
                elif xwoba >= 0.300:
                    xwoba_color = "#334155"   # dark gray
                else:
                    xwoba_color = "#94a3b8"   # muted
                xwoba_str = f'<span style="color:{xwoba_color};font-weight:600;">{xwoba:.3f}</span>'
            else:
                xwoba_str = '<span style="color:#cbd5e1;">—</span>'

            ops_str = f"{ops:.3f}" if ops else "—"
            bg = "background:#fafafa;" if i % 2 == 0 else ""

            rows.append(
                f'<tr style="{bg}">'
                f'<td style="padding:4px 6px;font-size:11px;color:#94a3b8;">{order}</td>'
                f'<td style="padding:4px 6px;font-size:11px;color:#1e293b;font-weight:500;">{name}</td>'
                f'<td style="padding:4px 6px;font-size:10px;color:#64748b;text-align:center;">{pos}</td>'
                f'<td style="padding:4px 6px;font-size:11px;text-align:right;">{xwoba_str}</td>'
                f'<td style="padding:4px 6px;font-size:11px;color:#475569;text-align:right;">{ops_str}</td>'
                f'</tr>'
            )
        return "\n".join(rows)

    away_confirmed = any(getattr(b, "is_confirmed", False) for b in away_lineup) if away_lineup else False
    home_confirmed = any(getattr(b, "is_confirmed", False) for b in home_lineup) if home_lineup else False

    away_rows = _team_rows(away_lineup, away_confirmed)
    home_rows = _team_rows(home_lineup, home_confirmed)

    return f"""
        <tr>
          <td style="padding:0 20px 16px 20px;border-top:1px solid #f1f5f9;">
            <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;">
              <tr>
                <!-- Away lineup -->
                <td width="50%" style="vertical-align:top;padding-right:10px;">
                  <div style="font-size:13px;font-weight:700;color:#1e293b;margin-bottom:6px;">{away_short}</div>
                  <table width="100%" cellpadding="0" cellspacing="0"
                         style="border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">
                    {away_rows}
                  </table>
                </td>
                <!-- Home lineup -->
                <td width="50%" style="vertical-align:top;padding-left:10px;">
                  <div style="font-size:13px;font-weight:700;color:#1e293b;margin-bottom:6px;">{home_short}</div>
                  <table width="100%" cellpadding="0" cellspacing="0"
                         style="border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">
                    {home_rows}
                  </table>
                </td>
              </tr>
            </table>
          </td>
        </tr>"""


def _send_via_resend(
    api_key:    str,
    from_addr:  str,
    recipients: list,
    subject:    str,
    html:       str,
) -> bool:
    """Low-level Resend send. Returns True on success."""
    try:
        import resend  # type: ignore
        resend.api_key = api_key
        response = resend.Emails.send({
            "from":    from_addr,
            "to":      recipients,
            "subject": subject,
            "html":    html,
        })
        email_id = (response.get("id", "unknown") if isinstance(response, dict)
                    else getattr(response, "id", "unknown"))
        logger.info("Email sent to %s | id=%s | subject=%r", recipients, email_id, subject)
        return True
    except ImportError:
        logger.warning("resend package not installed — run: pip install resend")
        return False
    except Exception as exc:
        logger.error("Resend email failed: %s", exc)
        return False
