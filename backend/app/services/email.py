"""Transactional email via Gmail SMTP (see Settings.SMTP_* in app/core/config.py — an
account-level App Password, not a general-purpose provider like Resend/SendGrid).
Fine at this app's current volume: Gmail's free-account cap is ~500 sends/day.

Called from a FastAPI BackgroundTask (see app/api/auth.py) so the SMTP round-trip
never adds latency to the request that triggered it.
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.config import settings

logger = logging.getLogger(__name__)


def send_email(*, to: str, subject: str, html_body: str) -> None:
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        # Local dev without SMTP configured shouldn't crash the request that
        # triggered the email — log and no-op instead.
        logger.warning("SMTP not configured — skipping email to %s (%s)", to, subject)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_USER}>"
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
    except Exception:
        # Best-effort: the OTP/notification is still valid even if the email fails to
        # send (e.g. a transient SMTP hiccup) — surface it in logs, not to the patient.
        logger.exception("Failed to send email to %s (%s)", to, subject)


_OTP_EMAIL_TEMPLATE = """\
<div style="font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px 24px;">
  <div style="font-size: 15px; font-weight: 700; letter-spacing: 0.02em; color: #3f6b3f; margin-bottom: 24px;">
    QuickCheck Clinic
  </div>
  <h1 style="font-size: 20px; margin: 0 0 12px; color: #14171a;">Reset your password</h1>
  <p style="font-size: 15px; line-height: 1.5; color: #3a3d42; margin: 0 0 24px;">
    Hi {full_name}, use the code below to reset your QuickCheck Clinic password. This
    code expires in {ttl_minutes} minutes.
  </p>
  <div style="font-size: 32px; font-weight: 700; letter-spacing: 0.3em; color: #14171a; background: #f0efe9; border-radius: 12px; padding: 16px 0; text-align: center; margin-bottom: 24px;">
    {otp_code}
  </div>
  <p style="font-size: 13px; line-height: 1.5; color: #6b6f76; margin: 0;">
    If you didn't request this, you can safely ignore this email — your password
    won't change unless this code is used.
  </p>
</div>
"""


def send_password_reset_otp_email(*, to: str, full_name: str, otp_code: str, ttl_minutes: int) -> None:
    html = _OTP_EMAIL_TEMPLATE.format(full_name=full_name, otp_code=otp_code, ttl_minutes=ttl_minutes)
    send_email(to=to, subject="Your QuickCheck Clinic verification code", html_body=html)
