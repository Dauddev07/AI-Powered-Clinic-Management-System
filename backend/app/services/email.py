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
import threading
from datetime import datetime, timezone

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def _send_in_background(target, /, *args, **kwargs) -> None:
    """Fires an email off a daemon thread rather than blocking the caller — used by
    the appointment lifecycle emails below, which (unlike the OTP/welcome emails)
    are triggered from app.services.booking_engine and app.services.
    appointment_reminders: shared service-layer code called both from a FastAPI
    request (which has BackgroundTasks available) AND from the chatbot's tool-calling
    path and the scheduler's background tick (neither of which do). A plain thread
    works uniformly in all three, at the cost of no built-in retry/backoff — same
    best-effort tradeoff send_email() below already makes for its own failures.
    """
    threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True).start()

_BREVO_SEND_URL = "https://api.brevo.com/v3/smtp/email"

_ACCENT = "#2563eb"
_ACCENT_DARK = "#1d4ed8"
_ACCENT_SOFT = "#eaf0fe"
_INK = "#0b1220"
_BODY_TEXT = "#33404f"
_MUTED = "#5b667a"
_BORDER = "#e4e9f5"
_SURFACE = "#f7f9fe"
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
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:520px;background:#ffffff;border:1px solid {_BORDER};border-radius:18px;overflow:hidden;box-shadow:0 12px 32px rgba(15,23,42,0.10);">
        <tr>
          <td style="height:5px;background:linear-gradient(90deg,{_ACCENT},{_ACCENT_DARK});font-size:0;line-height:0;">&nbsp;</td>
        </tr>
        <tr>
          <td align="center" style="padding:36px 32px 10px;background:linear-gradient(180deg,{_ACCENT_SOFT},#ffffff);">
            <table role="presentation" cellpadding="0" cellspacing="0" style="margin:0 auto 16px;">
              <tr>
                <td width="56" height="56" align="center" valign="middle" style="width:56px;height:56px;border-radius:50%;background:#ffffff;border:1.5px solid {_ACCENT_SOFT};box-shadow:0 4px 12px rgba(15,23,42,0.08);font-size:25px;line-height:56px;">
                  {icon}
                </td>
              </tr>
            </table>
            <div style="font-family:{_FONT};font-size:13px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{_ACCENT};">
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
        <td align="center" style="background:linear-gradient(180deg,{_ACCENT_SOFT},#ffffff);border:1.5px solid {_ACCENT_SOFT};border-radius:14px;padding:24px 0;box-shadow:inset 0 1px 0 #ffffff;">
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


def send_email_verification_otp_email(*, to: str, full_name: str, otp_code: str, ttl_minutes: int) -> None:
    body = f"""\
    <p style="margin:0 0 22px;text-align:center;">
      Hi {full_name}, enter the code below to verify your email and finish setting
      up your QuickCheck Clinic account. This code expires in
      <strong>{ttl_minutes} minutes</strong>.
    </p>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 18px;">
      <tr>
        <td align="center" style="background:linear-gradient(180deg,{_ACCENT_SOFT},#ffffff);border:1.5px solid {_ACCENT_SOFT};border-radius:14px;padding:24px 0;box-shadow:inset 0 1px 0 #ffffff;">
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
      If you didn't create this account, you can safely ignore this email.
    </p>
    """
    html = _layout(
        preheader=f"Your verification code is {otp_code} — expires in {ttl_minutes} minutes.",
        icon="✉️",
        heading="Verify your email",
        body_html=body,
    )
    send_email(to=to, subject="Verify your email for QuickCheck Clinic", html_body=html)


def _feature_row(*, icon: str, text: str) -> str:
    return f"""\
    <tr>
      <td style="padding:0 0 10px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_SURFACE};border-radius:12px;">
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
                    "<strong>Describe your symptoms to Cura</strong> — it asks a "
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


def _detail_row(*, label: str, value: str) -> str:
    return f"""\
    <tr>
      <td style="padding:6px 0;font-size:13px;color:{_MUTED};width:110px;">{label}</td>
      <td style="padding:6px 0;font-size:14px;color:{_INK};font-weight:600;">{value}</td>
    </tr>
    """


def send_appointment_booked_email(
    *, to: str, full_name: str, doctor_name: str, department_name: str, when_text: str
) -> None:
    def _send() -> None:
        body = f"""\
        <p style="margin:0 0 20px;text-align:center;">
          Hi {full_name}, your appointment is confirmed.
        </p>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;background:{_SURFACE};border-radius:14px;padding:16px 20px;">
          {_detail_row(label="Doctor", value=doctor_name)}
          {_detail_row(label="Department", value=department_name)}
          {_detail_row(label="When", value=when_text)}
        </table>
        <p style="margin:0;font-size:13px;color:{_MUTED};border-top:1px solid {_BORDER};padding-top:16px;">
          Need to make a change? You can reschedule or cancel anytime from your
          Upcoming Appointments page.
        </p>
        """
        html = _layout(
            preheader=f"Confirmed: {doctor_name} on {when_text}.",
            icon="✅",
            heading="Appointment confirmed",
            body_html=body,
        )
        send_email(to=to, subject="Your QuickCheck Clinic appointment is confirmed", html_body=html)

    _send_in_background(_send)


def send_appointment_cancelled_email(*, to: str, full_name: str, doctor_name: str, when_text: str) -> None:
    def _send() -> None:
        body = f"""\
        <p style="margin:0 0 20px;text-align:center;">
          Hi {full_name}, your appointment has been cancelled.
        </p>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;background:{_SURFACE};border-radius:14px;padding:16px 20px;">
          {_detail_row(label="Doctor", value=doctor_name)}
          {_detail_row(label="Was set for", value=when_text)}
        </table>
        <p style="margin:0;font-size:13px;color:{_MUTED};border-top:1px solid {_BORDER};padding-top:16px;">
          Changed your mind? You can book a new appointment anytime from Book Appointment.
        </p>
        """
        html = _layout(
            preheader=f"Cancelled: {doctor_name} on {when_text}.",
            icon="✖️",
            heading="Appointment cancelled",
            body_html=body,
        )
        send_email(to=to, subject="Your QuickCheck Clinic appointment was cancelled", html_body=html)

    _send_in_background(_send)


def send_appointment_rescheduled_email(
    *,
    to: str,
    full_name: str,
    old_doctor_name: str,
    old_when_text: str,
    new_doctor_name: str,
    new_when_text: str,
) -> None:
    def _send() -> None:
        # Doctor name is always shown alongside the time, even when it's the same
        # doctor for both — a patient scanning just the From/To rows shouldn't have
        # to infer "same doctor" from its absence.
        rows = (
            _detail_row(label="From", value=f"{old_doctor_name} — {old_when_text}")
            + _detail_row(label="To", value=f"{new_doctor_name} — {new_when_text}")
        )
        body = f"""\
        <p style="margin:0 0 20px;text-align:center;">
          Hi {full_name}, your appointment has been rescheduled.
        </p>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;background:{_SURFACE};border-radius:14px;padding:16px 20px;">
          {rows}
        </table>
        <p style="margin:0;font-size:13px;color:{_MUTED};border-top:1px solid {_BORDER};padding-top:16px;">
          Need to make another change? You can reschedule or cancel anytime from your
          Upcoming Appointments page.
        </p>
        """
        html = _layout(
            preheader=f"Rescheduled to {new_when_text}.",
            icon="🔁",
            heading="Appointment rescheduled",
            body_html=body,
        )
        send_email(to=to, subject="Your QuickCheck Clinic appointment was rescheduled", html_body=html)

    _send_in_background(_send)


def send_appointment_reminder_email(*, to: str, full_name: str, doctor_name: str, when_text: str) -> None:
    def _send() -> None:
        body = f"""\
        <p style="margin:0 0 20px;text-align:center;">
          Hi {full_name}, this is a reminder that your appointment with
          <strong>{doctor_name}</strong> is coming up in about <strong>1 hour</strong>.
        </p>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;background:{_SURFACE};border-radius:14px;padding:16px 20px;">
          {_detail_row(label="Doctor", value=doctor_name)}
          {_detail_row(label="When", value=when_text)}
        </table>
        <p style="margin:0;font-size:13px;color:{_MUTED};border-top:1px solid {_BORDER};padding-top:16px;">
          Appointments can't be cancelled or rescheduled within 2 hours of the appointment
          time. If you can't make it, please contact the clinic directly.
        </p>
        """
        html = _layout(
            preheader=f"Reminder: {doctor_name} in about 1 hour ({when_text}).",
            icon="⏰",
            heading="Your appointment is in 1 hour",
            body_html=body,
        )
        send_email(to=to, subject="Reminder: your QuickCheck Clinic appointment is in 1 hour", html_body=html)

    _send_in_background(_send)
