"""
GraphoLab core — Document Dating.

Provides:
  - extract_dates()   extract and normalize dates from OCR text
  - dating_rank()     rank a list of document file paths by date
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Date extraction patterns (Italian)
# ──────────────────────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    # "10 gennaio 2024" / "10 gennaio del 2024"
    r"\b(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|"
    r"luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(?:del\s+)?(\d{4})\b",
    # "10 gen. 2024" abbreviations
    r"\b(\d{1,2})\s+(gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\.?\s+(\d{4})\b",
    # "10/01/2024" or "10-01-2024" or "10.01.2024"
    r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b",
    # "gennaio 2024" (no day)
    r"\b(gennaio|febbraio|marzo|aprile|maggio|giugno|"
    r"luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4})\b",
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)

_BIRTH_KW = ("nata", "nato", "nascita", "nasc.", "nata il", "nato il")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_dateparser(raw: str) -> datetime | None:
    """Parse a raw date string to datetime using dateparser (Italian-aware)."""
    try:
        import dateparser
        dt = dateparser.parse(
            raw,
            languages=["it", "en"],
            settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if dt and 1800 < dt.year < 2200:
            return dt
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_dates(text: str) -> list[tuple[str, datetime]]:
    """Extract and normalize dates from OCR text.

    Returns a list of (raw_string, datetime) pairs sorted chronologically.
    Uses regex first; falls back to scanning NER DATE entities if nothing found.
    """
    found: list[tuple[str, datetime]] = []

    for m in _DATE_RE.finditer(text):
        raw = m.group(0).strip()
        context_before = text[max(0, m.start() - 35): m.start()].lower()
        if any(kw in context_before for kw in _BIRTH_KW):
            continue
        dt = _try_dateparser(raw)
        if dt:
            found.append((raw, dt))

    # NER fallback
    if not found:
        try:
            from core.ner import ner_extract
            _, ner_md = ner_extract(text)
            for raw in re.findall(r"\*\*([^*]+)\*\*\s*`DATE`", ner_md or ""):
                dt = _try_dateparser(raw)
                if dt:
                    found.append((raw, dt))
        except Exception:
            pass

    # De-duplicate by normalized date
    seen: set[str] = set()
    unique: list[tuple[str, datetime]] = []
    for raw, dt in found:
        key = dt.strftime("%Y-%m-%d")
        if key not in seen:
            seen.add(key)
            unique.append((raw, dt))

    return sorted(unique, key=lambda x: x[1])


def dating_rank(file_paths: list[str | Path]) -> str:
    """Rank documents by extracted date.

    Args:
        file_paths: List of image file paths (strings or Path objects).

    Returns:
        Markdown table with documents sorted chronologically.
    """
    if not file_paths:
        return "Carica almeno un'immagine di documento."

    from core.ocr import get_easyocr
    reader = get_easyocr()
    rows: list[tuple[str, str, datetime | None]] = []

    for fp in file_paths:
        path = Path(fp)
        name = path.name
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
            ocr_lines = reader.readtext(img_np, detail=0, paragraph=False)
            text = "\n".join(ocr_lines)
            dates = extract_dates(text)
            if dates:
                raw, dt = dates[-1]   # most recent date = document date
                rows.append((name, raw, dt))
            else:
                rows.append((name, "—  data non trovata", None))
        except Exception as e:
            rows.append((name, f"Errore: {e}", None))

    dated = [(n, r, dt) for n, r, dt in rows if dt is not None]
    undated = [(n, r, dt) for n, r, dt in rows if dt is None]
    dated.sort(key=lambda x: x[2])
    sorted_rows = dated + undated

    lines = [
        "## Datazione Documenti — Risultati\n",
        "| # | Documento | Data estratta | Data normalizzata |",
        "|---|-----------|--------------|-------------------|",
    ]
    for i, (name, raw, dt) in enumerate(sorted_rows, 1):
        norm = dt.strftime("%Y-%m-%d") if dt else "—"
        lines.append(f"| {i} | `{name}` | {raw} | {norm} |")

    if not dated:
        lines.append("\n> Nessuna data rilevata nei documenti caricati.")
    else:
        lines.append(
            f"\n*{len(dated)} document{'o' if len(dated)==1 else 'i'} datato/i, "
            f"{len(undated)} senza data.*"
        )

    return "\n".join(lines)
