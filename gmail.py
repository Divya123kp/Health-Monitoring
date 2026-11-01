"""
gmail.py — Gmail helpers (SMTP send / IMAP read).

Based on the supplied gmail.py. Changes:
  * credentials come from .env (GMAIL_SENDER / GMAIL_APP_PASSWORD), not source code
  * attachment sender removed (alerts are plain text only)
  * the test e-mails that used to run at import time were removed
"""
import email
import imaplib
import os
import smtplib
from email.header import decode_header
from email.message import EmailMessage

import config  # noqa: F401  (loads .env)


def send_email(from_email_addr, from_email_pass, to_email_addr, subject, body):
    msg = EmailMessage()
    msg.set_content(body)                      # plain text, UTF-8 (Kannada safe)
    msg["From"] = from_email_addr
    msg["To"] = to_email_addr
    msg["Subject"] = subject

    server = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
    try:
        server.starttls()
        server.login(from_email_addr, from_email_pass)
        server.send_message(msg)
    finally:
        server.quit()


def send_from_env(to_email_addr, subject, body):
    """send_email() using the sender configured in .env"""
    sender = os.getenv("GMAIL_SENDER", "")
    password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not sender or not password:
        raise RuntimeError("GMAIL_SENDER / GMAIL_APP_PASSWORD not set in .env")
    send_email(sender, password, to_email_addr, subject, body)


def get_email(folder_name="INBOX"):
    """Print every mail in a folder (reads with GMAIL_SENDER / GMAIL_APP_PASSWORD)."""
    email_user = os.getenv("GMAIL_SENDER", "")
    email_pass = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(email_user, email_pass)
    mail.select(folder_name)
    status, messages = mail.search(None, "ALL")
    for msg_id in messages[0].split():
        _, msg_data = mail.fetch(msg_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        subject, _enc = decode_header(msg["Subject"])[0]
        sender, _enc = decode_header(msg.get("From"))[0]
        print(f"Subject: {subject}\nFrom: {sender}")
        if msg.is_multipart():
            for part in msg.walk():
                disp = str(part.get("Content-Disposition"))
                if part.get_content_type() == "text/plain" and "attachment" not in disp:
                    print(part.get_payload(decode=True).decode("utf-8", "replace"))
        else:
            print(msg.get_payload(decode=True).decode("utf-8", "replace"))
        print("-" * 50)
    mail.logout()
