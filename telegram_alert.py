import os

import config  # noqa: F401  (loads .env)

# Set in .env (or real environment variables):  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
_bot = None


def _get_bot(token):
    global _bot
    if _bot is None:
        import telepot
        _bot = telepot.Bot(token)
    return _bot


def send_alert(msg: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - alert not sent")
        return False
    try:
        try:
            _get_bot(token).sendMessage(chat_id, str(msg))
        except ImportError:
            # telepot is unmaintained and may not install on new Python versions:
            # fall back to the plain Telegram HTTPS API.
            import json, urllib.request
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=json.dumps({"chat_id": chat_id, "text": str(msg)}).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15).read()
        print("[telegram] sent")
        return True
    except Exception as e:
        print(f"[telegram] FAILED: {e}")
        return False
