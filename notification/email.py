"""
Apex Analytics — Email Notification via Resend API
Sends the HTML report to configured recipients.

Requires:
  RESEND_API_KEY   — from resend.com (free: 3,000 emails/month)
  REPORT_EMAIL_TO  — recipient email address
  REPORT_EMAIL_FROM — sender address (must be verified domain in Resend)

Silently skips if RESEND_API_KEY is not set (not a fatal error).
"""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def send_report_email(
    report_html_path: str,
    game_date:        date,
    subject_suffix:   str = "",
    recipients:       Optional[list] = None,
) -> bool:
    """
    Send the HTML report via Resend.

    Parameters
    ----------
    report_html_path : Absolute path to the generated HTML report file.
    game_date        : Date of the report (used in subject line).
    subject_suffix   : Optional suffix for the subject line (e.g., "Pre-Game Update").
    recipients       : Override recipient list (defaults to REPORT_EMAIL_TO env var).

    Returns
    -------
    bool — True on success, False if skipped or failed.
    """
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        logger.debug("RESEND_API_KEY not set — email notification skipped.")
        return False

    # Determine recipients
    if recipients is None:
        to_env = os.environ.get("REPORT_EMAIL_TO", "").strip()
        if not to_env:
            logger.warning("REPORT_EMAIL_TO not set — email skipped.")
            return False
        recipients = [addr.strip() for addr in to_env.split(",") if addr.strip()]

    from_addr = os.environ.get(
        "REPORT_EMAIL_FROM",
        "apex@yourdomain.com"
    ).strip()

    # Read HTML content
    try:
        html_content = Path(report_html_path).read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Could not read report file %s: %s", report_html_path, exc)
        return False

    # Build subject
    date_str = game_date.strftime("%B %-d, %Y")
    n_games  = html_content.count("class=\"game-card\"")
    subject  = f"Apex Analytics — {date_str} MLB Report ({n_games} games)"
    if subject_suffix:
        subject += f" — {subject_suffix}"

    # Send via Resend
    try:
        import resend  # type: ignore
        resend.api_key = api_key

        response = resend.Emails.send({
            "from":    from_addr,
            "to":      recipients,
            "subject": subject,
            "html":    html_content,
        })

        email_id = response.get("id", "unknown") if isinstance(response, dict) else getattr(response, "id", "unknown")
        logger.info(
            "Email sent to %s | id=%s | subject=%r",
            recipients, email_id, subject,
        )
        return True

    except ImportError:
        logger.warning(
            "resend package not installed — run: pip install resend\n"
            "Email notification skipped."
        )
        return False

    except Exception as exc:
        logger.error("Resend email failed: %s", exc)
        return False


def send_test_email(recipient: str) -> bool:
    """
    Send a simple test email to verify Resend configuration.
    Useful for initial setup verification.

    Usage:
      python -c "from notification.email import send_test_email; send_test_email('you@email.com')"
    """
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RESEND_API_KEY not set in environment.")
        return False

    from_addr = os.environ.get("REPORT_EMAIL_FROM", "apex@yourdomain.com")

    try:
        import resend
        resend.api_key = api_key
        response = resend.Emails.send({
            "from":    from_addr,
            "to":      [recipient],
            "subject": "Apex Analytics — Test Email",
            "html":    (
                "<h2 style='font-family:sans-serif'>Apex Analytics</h2>"
                "<p style='font-family:sans-serif'>Test email sent successfully. "
                "Your Resend configuration is working.</p>"
            ),
        })
        print(f"✓ Test email sent to {recipient}")
        return True
    except Exception as exc:
        print(f"✗ Test email failed: {exc}")
        return False
