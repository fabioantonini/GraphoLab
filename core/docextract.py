"""
GraphoLab core — Landing.ai Agentic Document Extraction (ADE).

Provides a thin, framework-agnostic wrapper around the ``landingai-ade`` SDK:
  - ade_parse()     → convert document/image to structured markdown
  - ade_extract()   → extract key/value fields via a Pydantic schema
  - ade_pipeline()  → parse + extract in one call

Pre-built forensic schemas (Pydantic v2):
  - ForensicReportSchema   — 20 ENFSI BPM fields
  - SignatureDocumentSchema — signature, date, parties
  - LegalActSchema         — deed, notary, signatories

Authentication: reads VISION_AGENT_API_KEY from the environment (same .env used
by the rest of GraphoLab). If the key is absent all functions raise a clear
RuntimeError so the Gradio tab can show a configuration message instead of crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Key management
# ──────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("VISION_AGENT_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "VISION_AGENT_API_KEY non configurata. "
            "Aggiungi la chiave nel file .env oppure nella sezione Configurazione."
        )
    return key


def ade_key_configured() -> bool:
    """Return True if VISION_AGENT_API_KEY is available in the environment."""
    return bool(os.environ.get("VISION_AGENT_API_KEY", "").strip())


# ──────────────────────────────────────────────────────────────────────────────
# Return-value container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AdeResult:
    """Aggregated result of an ADE parse + extract pipeline."""
    markdown: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)        # raw chunk metadata from parse
    extraction_metadata: dict[str, Any] = field(default_factory=dict)  # visual grounding refs
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ──────────────────────────────────────────────────────────────────────────────
# Core ADE operations
# ──────────────────────────────────────────────────────────────────────────────

def ade_parse(
    file_path: str | Path,
    model: str = "dpt-2-latest",
) -> tuple[str, list[dict]]:
    """Parse a document/image into structured markdown.

    Args:
        file_path: Path to the document (PDF, PNG, JPG, TIFF, …).
        model:     ADE model identifier (default: ``dpt-2-latest``).

    Returns:
        ``(markdown_text, chunks)`` where *chunks* is a list of raw metadata
        dicts (each chunk has location, type, page, etc.).

    Raises:
        RuntimeError: if VISION_AGENT_API_KEY is not configured.
        FileNotFoundError: if *file_path* does not exist.
    """
    _get_api_key()
    from landingai_ade import LandingAIADE

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")

    client = LandingAIADE()
    response = client.parse(document=file_path, model=model)
    markdown = response.markdown or ""
    chunks = [c.model_dump() if hasattr(c, "model_dump") else dict(c)
              for c in (response.chunks or [])]
    return markdown, chunks


def ade_extract(
    markdown: str,
    schema: Type[BaseModel] | str,
    model: str = "extract-latest",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract structured fields from ADE markdown using a Pydantic schema.

    Args:
        markdown: Markdown text produced by :func:`ade_parse`.
        schema:   A Pydantic ``BaseModel`` subclass **or** a pre-serialised JSON
                  schema string (as returned by ``pydantic_to_json_schema()``).
        model:    Extract model version (default: ``"extract-latest"``).

    Returns:
        ``(fields, extraction_metadata)`` where *fields* maps field names to
        extracted values and *extraction_metadata* maps field names to visual
        grounding references (chunk/cell IDs linking back to parse output).

    Raises:
        RuntimeError: if VISION_AGENT_API_KEY is not configured.
    """
    _get_api_key()
    from landingai_ade import LandingAIADE
    from landingai_ade.lib.schema_utils import pydantic_to_json_schema

    client = LandingAIADE()

    # schema must be a JSON string — use the SDK helper for Pydantic models
    if isinstance(schema, str):
        schema_str = schema
    else:
        schema_str = pydantic_to_json_schema(schema)

    response = client.extract(schema=schema_str, markdown=markdown, model=model)

    # Normalise response to plain dict
    if hasattr(response, "model_dump"):
        raw = response.model_dump()
    elif isinstance(response, dict):
        raw = response
    else:
        raw = dict(response)

    # The API returns {"extraction": {...}, "extraction_metadata": {...}, "metadata": {...}}
    fields = raw.get("extraction", raw)
    extraction_metadata = raw.get("extraction_metadata", {})
    return fields, extraction_metadata


def ade_pipeline(
    file_path: str | Path,
    schema: Type[BaseModel],
    model: str = "dpt-2-latest",
) -> AdeResult:
    """Parse a document and extract structured fields in one call.

    Convenience wrapper that chains :func:`ade_parse` → :func:`ade_extract`.

    Returns:
        :class:`AdeResult` with ``markdown``, ``fields``, ``chunks``.
        On error, ``AdeResult.error`` is set and ``AdeResult.ok`` is ``False``.
    """
    try:
        markdown, chunks = ade_parse(file_path, model=model)
        fields, extraction_metadata = ade_extract(markdown, schema)
        return AdeResult(
            markdown=markdown,
            fields=fields,
            chunks=chunks,
            extraction_metadata=extraction_metadata,
        )
    except Exception as exc:
        return AdeResult(error=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Pre-built forensic Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class ForensicReportSchema(BaseModel):
    """ENFSI BPM-FHX-01 Ed.03 — 20 required elements for a forensic report."""
    caso_identificatore: Optional[str] = Field(None, description="Identificatore univoco del caso (numero fascicolo, repertorio)")
    laboratorio_nome: Optional[str] = Field(None, description="Nome del laboratorio o professionista")
    laboratorio_indirizzo: Optional[str] = Field(None, description="Indirizzo o recapito del laboratorio")
    esaminatore_nome: Optional[str] = Field(None, description="Nome del perito esaminatore")
    esaminatore_qualifiche: Optional[str] = Field(None, description="Qualifiche, titoli, iscrizioni albo del perito")
    firma_esaminatore: Optional[str] = Field(None, description="Presenza della firma del perito (sì/no/autografa/digitale)")
    data_firma_relazione: Optional[str] = Field(None, description="Data di firma della relazione peritale")
    data_ricezione_materiale: Optional[str] = Field(None, description="Data di ricezione del materiale esaminato")
    trasmittente_nome: Optional[str] = Field(None, description="Nome di chi ha trasmesso il materiale")
    trasmittente_qualifica: Optional[str] = Field(None, description="Qualifica del trasmittente (avvocato, notaio, tribunale)")
    elenco_materiale: Optional[str] = Field(None, description="Elenco dei documenti/reperti esaminati")
    stato_materiale_ricezione: Optional[str] = Field(None, description="Condizioni del materiale alla ricezione")
    contesto_informativo: Optional[str] = Field(None, description="Informazioni di contesto ricevute con il materiale")
    scopo_esame: Optional[str] = Field(None, description="Quesito peritale / scopo dell'esame")
    metodi_esame: Optional[str] = Field(None, description="Metodi e tecniche di analisi applicati")
    risultati_esame: Optional[str] = Field(None, description="Risultati concreti delle analisi grafologiche")
    valutazione_risultati: Optional[str] = Field(None, description="Valutazione del peso e significato dei risultati")
    opinione_esperto: Optional[str] = Field(None, description="Opinione conclusiva del perito")
    limitazioni_metodologiche: Optional[str] = Field(None, description="Limiti e riserve metodologiche dichiarati")
    riferimenti_bibliografici: Optional[str] = Field(None, description="Fonti bibliografiche o normative citate")


class SignatureDocumentSchema(BaseModel):
    """Document containing a signature — parties, date, type."""
    tipo_documento: Optional[str] = Field(None, description="Tipo di documento (testamento, contratto, atto, procura, …)")
    data_documento: Optional[str] = Field(None, description="Data del documento")
    luogo_documento: Optional[str] = Field(None, description="Luogo di redazione del documento")
    firmatario_principale: Optional[str] = Field(None, description="Nome del firmatario principale")
    altri_firmatari: Optional[list[str]] = Field(None, description="Nomi degli altri firmatari o testimoni")
    notaio: Optional[str] = Field(None, description="Nome del notaio rogante (se presente)")
    numero_repertorio: Optional[str] = Field(None, description="Numero di repertorio notarile")
    oggetto_documento: Optional[str] = Field(None, description="Oggetto o causa del documento")
    presenza_firma: Optional[str] = Field(None, description="Descrizione della firma (autografa, olografa, a stampa, …)")
    note: Optional[str] = Field(None, description="Altre annotazioni rilevanti")


class LegalActSchema(BaseModel):
    """Generic legal act — deed, parties, notary."""
    tipo_atto: Optional[str] = Field(None, description="Tipo di atto (testamento, contratto, sentenza, ordinanza, …)")
    numero_atto: Optional[str] = Field(None, description="Numero identificativo dell'atto")
    data_atto: Optional[str] = Field(None, description="Data dell'atto")
    autorita_emittente: Optional[str] = Field(None, description="Autorità o ente che ha emesso l'atto")
    parti_coinvolte: Optional[list[str]] = Field(None, description="Parti coinvolte nell'atto (nomi)")
    rappresentanti_legali: Optional[list[str]] = Field(None, description="Avvocati o rappresentanti legali citati")
    oggetto_atto: Optional[str] = Field(None, description="Oggetto dell'atto in sintesi")
    importo: Optional[str] = Field(None, description="Importo economico se presente (es. valore immobile, patrimonio)")
    scadenze: Optional[str] = Field(None, description="Date o scadenze rilevanti menzionate")
    riferimenti_normativi: Optional[str] = Field(None, description="Norme di legge o articoli citati")


# Registry used by the Gradio tab to populate the schema dropdown
SCHEMA_REGISTRY: dict[str, Type[BaseModel]] = {
    "Perizia Forense (ENFSI)": ForensicReportSchema,
    "Documento con Firma": SignatureDocumentSchema,
    "Atto Legale": LegalActSchema,
}
