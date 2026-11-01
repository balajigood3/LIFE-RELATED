"""Optional OCR helpers for scanned documents / nameplates."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

def ocr_image(path: Path) -> str:
    """Best-effort OCR. Returns empty string if dependencies missing."""
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        return f"[OCR unavailable: {e}]"
