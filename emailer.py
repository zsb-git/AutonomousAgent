"""Send agent output via email, using Gmail's SMTP server.

Gmail requires an "App Password" for this (not your regular account
password) - generate one at myaccount.google.com/apppasswords, which
requires 2-Step Verification to be turned on first. Store it in .env as
GMAIL_APP_PASSWORD, alongside GMAIL_ADDRESS for the sending account.
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

# Loaded here (not just by the calling script) so this module gives the
# right answer regardless of import order relative to the caller's own
# load_dotenv() call.
load_dotenv()

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")


def send_email(subject: str, body: str, to_address: str, attachment_path: Path | None = None) -> None:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("Skipping email: GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set in .env")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_address
    msg.set_content(body)

    if attachment_path and attachment_path.exists():
        msg.add_attachment(
            attachment_path.read_bytes(),
            maintype="text",
            subtype="plain",
            filename=attachment_path.name,
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)

    print(f"Emailed output to {to_address}")
