"""
utils.py
--------
Chhoti chhoti helper functions jo poore project mein reuse hoti hain.
"""

import uuid
import re
from datetime import datetime


def generate_id(prefix: str = "CAND") -> str:
    """Unique candidate/report ID generate karta hai."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def current_timestamp() -> str:
    """Current timestamp string format mein return karta hai."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(text: str) -> str:
    """Text ko lowercase aur extra whitespace se saaf karta hai."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,+#-]", " ", text)
    return text.strip()


def safe_filename(name: str) -> str:
    """File name se special characters hata kar safe naam deta hai."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)
