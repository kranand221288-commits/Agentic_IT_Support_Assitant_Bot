"""
Utility helpers for the app.
"""
import html

def sanitize_text_for_display(text: str) -> str:
    return html.escape(text)

def short_preview(text: str, length: int = 200) -> str:
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[:length].rstrip() + "..."
