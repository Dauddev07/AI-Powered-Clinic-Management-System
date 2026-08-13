"""Transactional email via Brevo's HTTP API (see Settings.BREVO_* in
app/core/config.py). Not raw SMTP — Render blocks outbound SMTP ports on its
free/starter tiers (spam-abuse prevention), so smtplib works locally but fails in
production with "Network is unreachable". Brevo's API runs over plain HTTPS, which
isn't blocked, and its free tier (300 emails/day) needs no domain — only the sender
email verified in Brevo's own dashboard.

Called from a FastAPI BackgroundTask (see app/api/auth.py) so the HTTP round-trip
never adds latency to the request that triggered it.
"""
import logging
from datetime import datetime, timezone

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_BREVO_SEND_URL = "https://api.brevo.com/v3/smtp/email"

_ACCENT = "#3f6b3f"
_ACCENT_DARK = "#2f5330"
_ACCENT_SOFT = "#e2f0e0"
_INK = "#14171a"
_BODY_TEXT = "#3a3d42"
_MUTED = "#6b6f76"
_BORDER = "#e3e1da"
_SURFACE = "#f5f5f0"
_FONT = "-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif"


def send_email(*, to: str, subject: str, html_body: str) -> None:
    if not settings.BREVO_API_KEY or not settings.BREVO_SENDER_EMAIL:
        # Local dev without Brevo configured shouldn't crash the request that
        # triggered the email — log and no-op instead.
        logger.warning("Brevo not configured — skipping email to %s (%s)", to, subject)
        return

    try:
        response = httpx.post(
            _BREVO_SEND_URL,
            headers={
                "api-key": settings.BREVO_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "sender": {"name": settings.BREVO_SENDER_NAME, "email": settings.BREVO_SENDER_EMAIL},
                "to": [{"email": to}],
                "subject": subject,
                "htmlContent": html_body,
            },
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        # Best-effort: the OTP/notification is still valid even if the email fails to
        # send (e.g. a transient API hiccup) — surface it in logs, not to the patient.
        logger.exception("Failed to send email to %s (%s)", to, subject)


# Shared table-based layout (not just a styled <div>) — Gmail/most modern clients
# render inline-styled divs fine, but Outlook and some older/enterprise clients only
# reliably respect table-based layout, so every real template here is built on one.
# preheader is the (visually hidden) snippet text email clients show in the inbox
# list next to the subject — without it, clients fall back to grabbing the first
# visible text, which is normally the brand name repeated for every single email.
def _layout(*, preheader: str, icon: str, heading: str, body_html: str) -> str:
    year = datetime.now(timezone.utc).year
    return f"""\
<div style="display:none;max-height:0;overflow:hidden;opacity:0;">{preheader}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_SURFACE};padding:40px 16px;">
  <tr>
    <td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:520px;background:#ffffff;border:1px solid {_BORDER};border-radius:16px;overflow:hidden;box-shadow:0 8px 24px rgba(30,28,20,0.08);">
        <tr>
          <td style="height:5px;background:linear-gradient(90deg,{_ACCENT},{_ACCENT_DARK});font-size:0;line-height:0;">&nbsp;</td>
        </tr>
        <tr>
          <td align="center" style="padding:34px 32px 8px;background:linear-gradient(180deg,{_ACCENT_SOFT},#ffffff);">
            <table role="presentation" cellpadding="0" cellspacing="0" style="margin:0 auto 14px;">
              <tr>
                <td width="52" height="52" align="center" valign="middle" style="width:52px;height:52px;border-radius:50%;background:#ffffff;border:1px solid {_BORDER};font-size:24px;line-height:52px;">
                  {icon}
                </td>
              </tr>
            </table>
            <div style="font-family:{_FONT};font-size:14px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:{_ACCENT};">
              QuickCheck Clinic
            </div>
          </td>
        </tr>
        <tr>
          <td align="center" style="padding:18px 32px 4px;">
            <h1 style="font-family:{_FONT};font-size:22px;line-height:1.3;margin:0;color:{_INK};text-align:center;">
              {heading}
            </h1>
          </td>
        </tr>
        <tr>
          <td style="padding:12px 36px 30px;font-family:{_FONT};font-size:15px;line-height:1.65;color:{_BODY_TEXT};">
            {body_html}
          </td>
        </tr>
        <tr>
          <td style="padding:20px 32px;background:{_SURFACE};border-top:1px solid {_BORDER};font-family:{_FONT};font-size:12px;line-height:1.7;color:{_MUTED};text-align:center;">
            This is an automated message from QuickCheck Clinic — please don't reply directly to this email.
            <br />© {year} QuickCheck Clinic. All rights reserved.
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>
"""


def send_password_reset_otp_email(*, to: str, full_name: str, otp_code: str, ttl_minutes: int) -> None:
    body = f"""\
    <p style="margin:0 0 22px;text-align:center;">
      Hi {full_name}, use the verification code below to reset your QuickCheck Clinic
      password. This code expires in <strong>{ttl_minutes} minutes</strong>.
    </p>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 18px;">
      <tr>
        <td align="center" style="background:{_ACCENT_SOFT};border:1.5px dashed {_ACCENT};border-radius:12px;padding:22px 0;">
          <span style="font-family:'Courier New',monospace;font-size:32px;font-weight:700;letter-spacing:0.4em;color:{_ACCENT_DARK};">
            {otp_code}
          </span>
        </td>
      </tr>
    </table>
    <p style="margin:0 0 22px;text-align:center;font-size:12px;color:{_MUTED};">
      This code can only be used once.
    </p>
    <p style="margin:0;font-size:13px;color:{_MUTED};border-top:1px solid {_BORDER};padding-top:16px;">
      If you didn't request this, you can safely ignore this email — your password
      won't change unless this code is used.
    </p>
    """
    html = _layout(
        preheader=f"Your verification code is {otp_code} — expires in {ttl_minutes} minutes.",
        icon="🔒",
        heading="Reset your password",
        body_html=body,
    )
    send_email(to=to, subject="Your QuickCheck Clinic verification code", html_body=html)


def _feature_row(*, icon: str, text: str) -> str:
    return f"""\
    <tr>
      <td style="padding:0 0 10px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_SURFACE};border-radius:10px;">
          <tr>
            <td width="46" align="center" valign="middle" style="width:46px;padding:14px 0 14px 12px;">
              <table role="presentation" cellpadding="0" cellspacing="0">
                <tr>
                  <td width="32" height="32" align="center" valign="middle" style="width:32px;height:32px;border-radius:50%;background:{_ACCENT_SOFT};font-size:15px;line-height:32px;">
                    {icon}
                  </td>
                </tr>
              </table>
            </td>
            <td style="padding:14px 16px 14px 10px;font-size:14px;line-height:1.55;">{text}</td>
          </tr>
        </table>
      </td>
    </tr>
    """


def send_welcome_email(*, to: str, full_name: str) -> None:
    features = "".join(
        [
            _feature_row(
                icon="💬",
                text=(
                    "<strong>Describe your symptoms to our assistant</strong> — it asks a "
                    "few quick questions and points you to the right department, no "
                    "guessing which one to pick."
                ),
            ),
            _feature_row(
                icon="📅",
                text=(
                    "<strong>See real doctor availability</strong> and book a slot that "
                    "actually works for you, no back-and-forth calls."
                ),
            ),
            _feature_row(
                icon="🔁",
                text="<strong>Manage your visits</strong> — reschedule or cancel anytime, all in one place.",
            ),
        ]
    )
    body = f"""\
    <p style="margin:0 0 22px;text-align:center;">
      Hi {full_name}, your QuickCheck Clinic account has been created successfully.
      Here's what you can do now:
    </p>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 22px;">
      {features}
    </table>
    <p style="margin:0;font-size:13px;color:{_MUTED};border-top:1px solid {_BORDER};padding-top:16px;">
      If you didn't create this account, you can safely ignore this email.
    </p>
    """
    html = _layout(
        preheader="Your QuickCheck Clinic account is ready.",
        icon="👋",
        heading=f"Welcome, {full_name}",
        body_html=body,
    )
    send_email(to=to, subject="Welcome to QuickCheck Clinic", html_body=html)
