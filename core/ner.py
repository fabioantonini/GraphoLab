"""
GraphoLab core — Named Entity Recognition (NER).

Provides:
  - get_ner()       lazy loader for the NER pipeline
  - ner_extract()   extract named entities from text, returns structured result
"""

from __future__ import annotations

import os
import threading

from transformers import pipeline as hf_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

NER_MODEL = "Babelscape/wikineural-multilingual-ner"

_NER_LABELS = {
    "PER": "Persona",
    "ORG": "Organizzazione",
    "LOC": "Luogo",
    "MISC": "Varie",
}

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loader
# ──────────────────────────────────────────────────────────────────────────────

_ner_pipeline = None
_ner_lock = threading.Lock()


def get_ner():
    """Return the NER pipeline, loading it on first call (thread-safe)."""
    global _ner_pipeline
    if _ner_pipeline is None:
        with _ner_lock:
            if _ner_pipeline is None:
                import torch
                device = 0 if torch.cuda.is_available() else -1
                print("Loading NER model...")
                _ner_pipeline = hf_pipeline(
                    "ner",
                    model=NER_MODEL,
                    aggregation_strategy="simple",
                    device=device,
                )
    return _ner_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def ner_extract(text: str) -> tuple[list[tuple[str, str | None]], str]:
    """Extract named entities from *text*.

    Returns:
        highlighted: list of (span, label|None) suitable for Gradio HighlightedText
        summary_md:  Markdown table of detected entities
    """
    if not text or not text.strip():
        return [], "Inserisci del testo da analizzare."

    nlp = get_ner()
    entities = nlp(text)

    # Build HighlightedText format: list of (span, label|None)
    result: list[tuple[str, str | None]] = []
    prev_end = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        if start > prev_end:
            result.append((text[prev_end:start], None))
        result.append((text[start:end], ent["entity_group"]))
        prev_end = end
    if prev_end < len(text):
        result.append((text[prev_end:], None))

    # Summary Markdown table
    if entities:
        rows = "\n".join(
            f"| **{_NER_LABELS.get(e['entity_group'], e['entity_group'])}** "
            f"(`{e['entity_group']}`) | {e['word']} | {e['score']:.0%} |"
            for e in entities
        )
        summary_md = f"| Tipo | Entità | Confidenza |\n|------|--------|------------|\n{rows}"
    else:
        summary_md = "Nessuna entità trovata."

    return result, summary_md
