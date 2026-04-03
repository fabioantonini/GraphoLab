"""
GraphoLab core — ENFSI Perizia Compliance Checker.

Provides:
  - extract_perizia_text(path)   extract text from a PDF perizia
  - compliance_check_stream()    stream the ENFSI compliance analysis via Ollama
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Generator

import requests

# ──────────────────────────────────────────────────────────────────────────────
# ENFSI BPM Section 13.2 — required elements for a forensic report
# ──────────────────────────────────────────────────────────────────────────────

_ENFSI_CHECKLIST = [
    ("Identificatore univoco del caso",
     "La relazione riporta un numero o codice univoco che identifica il caso "
     "(es. numero di repertorio, numero di fascicolo, codice pratica)."),

    ("Nome e indirizzo del laboratorio / professionista",
     "La relazione indica il nome del perito o dello studio/laboratorio che ha eseguito l'esame, "
     "con almeno un indirizzo o recapito. Cerca nell'intestazione o nel frontespizio."),

    ("Identità e qualifiche dell'esaminatore forense",
     "La relazione indica nome, titolo professionale e qualifiche del perito "
     "(es. iscrizioni ad albi, certificazioni, titoli accademici). "
     "Cerca nell'intestazione, nel curriculum allegato o nella firma finale."),

    ("Firma dell'esaminatore forense",
     "La relazione reca la firma autografa o digitale del perito responsabile, "
     "solitamente in calce all'ultima pagina o alle conclusioni."),

    ("Data di firma della relazione",
     "La relazione riporta la data in cui il documento peritale è stato completato e firmato dal perito. "
     "ATTENZIONE: NON confondere con la data dell'incarico, la data di ricezione del materiale "
     "o la data del decesso del soggetto esaminato."),

    ("Data di ricevimento del materiale esaminato",
     "La relazione indica la data in cui il perito ha fisicamente ricevuto il materiale da esaminare. "
     "ATTENZIONE: NON confondere con la data dell'incarico, la data del decesso, "
     "la data di pubblicazione del testamento o altre date presenti nel documento."),

    ("Nome e status di chi ha trasmesso il materiale",
     "La relazione indica chi ha consegnato o inviato il materiale al perito: nome, qualifica "
     "(es. avvocato, notaio, tribunale) e, se disponibile, il metodo di trasmissione."),

    ("Elenco del materiale presentato",
     "La relazione contiene un elenco numerato o strutturato di tutti i documenti/reperti esaminati, "
     "con una descrizione sufficiente a identificarli univocamente (es. tipo di documento, data, provenienza). "
     "Cerca una sezione 'Elenco documenti', 'Materiale in esame' o simile."),

    ("Commento sullo stato del materiale alla ricezione",
     "La relazione descrive le condizioni fisiche del materiale al momento della ricezione "
     "(es. originale / fotocopia / scansione, stato di conservazione, integrità della busta). "
     "Una semplice menzione che il documento è stato esaminato in copia NON è sufficiente: "
     "occorre un commento esplicito sulle condizioni all'arrivo."),

    ("Informazioni di contesto ricevute con il materiale",
     "La relazione riporta le informazioni di background trasmesse insieme al materiale: "
     "quesito posto, contesto della controversia, dati biografici del soggetto esaminato, "
     "istruzioni particolari ricevute. Cerca nella sezione 'Incarico', 'Quesito' o 'Premessa'."),

    ("Scopo dell'esame",
     "La relazione dichiara in modo esplicito lo scopo forense: la domanda specifica a cui l'esame "
     "deve rispondere (es. 'verificare se il testamento è autografo'). "
     "Il quesito trascritto letteralmente soddisfa questo requisito."),

    ("Descrizione dei metodi e degli esami effettuati",
     "La relazione descrive i metodi grafologici, tecnici e strumentali applicati nell'analisi "
     "(es. analisi grafologica, confronto comparativo, microscopia, riflettografia). "
     "Cerca una sezione 'Metodologia', 'Metodo', 'Fasi di lavoro' o 'Strumentazione'. "
     "Una sezione dedicata con elenco di parametri grafici analizzati SODDISFA questo requisito."),

    ("Risultati dell'esame / analisi",
     "La relazione riporta i risultati concreti delle analisi: osservazioni grafiche, "
     "caratteristiche rilevate, confronti effettuati, tabelle comparative. "
     "Cerca le sezioni 'Esame', 'Analisi', 'Confronto', 'Fase confrontuale'."),

    ("Valutazione della significatività dei risultati",
     "La relazione valuta il peso e il significato dei risultati nel contesto del caso: "
     "spiega perché certe analogie o divergenze sono probanti o meno. "
     "Una sezione 'Bilanciamento analogie e divergenze' o 'Valutazione' con spiegazione "
     "del perché le differenze non siano ascrivibili a variazione naturale SODDISFA questo requisito."),

    ("Opinione dell'esperto e fattori che la influenzano",
     "La relazione contiene l'opinione conclusiva del perito e indica eventuali fattori "
     "che potrebbero condizionarla (es. esame eseguito solo su copia, mancanza di originale, "
     "assenza di documentazione clinica). Cerca le 'Conclusioni'."),

    ("Commento su materiale non esaminato",
     "Se parte del materiale non è stata esaminata o è stata esaminata in forma ridotta "
     "(es. solo fotocopia, senza l'originale), la relazione lo segnala con motivazione. "
     "Una nota che precisa 'esaminato solo in copia' e che si riserva l'ispezione sull'originale "
     "SODDISFA parzialmente questo requisito."),

    ("Sistema di numerazione delle pagine",
     "Le pagine sono numerate. Il formato 'pagina X di Y' è preferibile ma anche la sola "
     "numerazione progressiva (1, 2, 3…) è accettabile. "
     "Indica anche il totale pagine se presente (es. 'Tot pag. 27')."),

    ("Scala di certezza / livello di probabilità della conclusione",
     "Le conclusioni esprimono un livello di certezza: può essere la scala ENFSI formale "
     "(da 'certamente' a 'probabilmente non') o un'espressione equivalente come "
     "'con alta probabilità', 'con alta probabilità tecnica', 'con certezza'. "
     "Se la certezza è espressa in forma narrativa ma non con la scala ENFSI ufficiale, "
     "il requisito è PARZIALMENTE soddisfatto."),

    ("Limitazioni e riserve metodologiche",
     "La relazione menziona i limiti dell'analisi: es. esame su copia e non sull'originale, "
     "assenza di documentazione clinica, limitazioni dello strumento usato. "
     "Riserve esplicite su integrazioni future o su ispezioni non ancora effettuate "
     "SODDISFANO questo requisito."),

    ("Riferimenti bibliografici",
     "La relazione cita fonti bibliografiche o normative a supporto della metodologia: "
     "manuali, standard internazionali (es. ENFSI BPM), articoli scientifici. "
     "Citazioni in nota a piè di pagina o in una sezione dedicata SODDISFANO questo requisito "
     "anche se non è presente una bibliografia formale separata."),
]


# ──────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_perizia_text(path: Path) -> str:
    """Extract text from a PDF perizia using pypdf (+ EasyOCR fallback for scanned pages)."""
    from core.rag import _extract_pdf_text  # reuse existing implementation
    return _extract_pdf_text(path)


# ──────────────────────────────────────────────────────────────────────────────
# Compliance check (streaming)
# ──────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    System prompt: establishes the evaluator role and the exact output format.
    Kept separate from the perizia text so the model treats them as distinct inputs.
    """
    n = len(_ENFSI_CHECKLIST)
    checklist_lines = "\n".join(
        f"  REQ-{i+1:02d}. {name} — {desc}"
        for i, (name, desc) in enumerate(_ENFSI_CHECKLIST)
    )
    return f"""\
Sei un valutatore di conformità forense specializzato negli standard ENFSI.
Il tuo UNICO compito è verificare se una perizia grafologica rispetta i {n} requisiti
del Best Practice Manual ENFSI-BPM-FHX-01 Ed.03, sezione 13.2, elencati di seguito.

NON fare analisi grafologica. NON esprimere opinioni sul contenuto della perizia.
Valuta SOLO se ogni requisito formale è presente, parzialmente presente o assente.

════════ CHECKLIST REQUISITI ENFSI BPM sez. 13.2 ════════
{checklist_lines}
═════════════════════════════════════════════════════════

════════ FORMATO DI RISPOSTA — DUE FASI OBBLIGATORIE ════════

╔══ FASE 1 — VALUTAZIONE BLOCCO PER BLOCCO ══╗
Scrivi esattamente {n} blocchi, in ordine REQ-01 … REQ-{n:02d}. Formato fisso:

REQ-XX. [Nome requisito]
  Verdetto: [✅ CONFORME | ⚠️ PARZIALE | ❌ MANCANTE]
  Motivazione: [cita la sezione o le parole esatte della perizia; se assente spiega il perché]
  💡 Suggerimento: [SOLO se ⚠️ o ❌ — azione concreta che il perito deve compiere]

REGOLE DI COERENZA (violazioni = errore grave):
  • ✅ CONFORME  → la Motivazione DEVE indicare dove si trova il requisito nella perizia.
  • ⚠️ PARZIALE → la Motivazione DEVE specificare cosa c'è E cosa manca.
  • ❌ MANCANTE  → la Motivazione DEVE spiegare l'assenza.
  • VIETATO: usare ✅ e scrivere "assente" o "non trovato" nella motivazione.
  • VIETATO: usare ❌ e poi citare testo della perizia come se fosse presente.

Esempio corretto:
  REQ-01. Identificatore univoco del caso
    Verdetto: ✅ CONFORME
    Motivazione: La perizia riporta "rep.n.ro 4844 racc.n.ro 3443" nell'intestazione.

Rispondi SOLO in italiano. Inizia direttamente con REQ-01 senza preamboli.
Fermati dopo REQ-{n:02d}. NON aggiungere riepilogo, tabelle o commenti finali."""


def _build_user_prompt(perizia_text: str) -> str:
    """User prompt: just the perizia text, clearly delimited."""
    max_chars = 24_000  # ~6 000 tokens — fits comfortably in 16 k ctx
    if len(perizia_text) > max_chars:
        perizia_text = perizia_text[:max_chars] + "\n\n[... testo troncato per lunghezza ...]"
    return f"""\
/no_think
Valuta la seguente perizia rispetto alla checklist ENFSI BPM che ti ho fornito.

════════ TESTO DELLA PERIZIA ════════
{perizia_text}
═════════════════════════════════════

Produci ora esattamente i {len(_ENFSI_CHECKLIST)} blocchi REQ-01 … REQ-{len(_ENFSI_CHECKLIST):02d}."""


def _parse_verdicts(text: str) -> dict[int, str]:
    """Extract REQ-XX → symbol mappings from the LLM output."""
    verdicts: dict[int, str] = {}
    # Match "REQ-XX" (with optional bold/heading markers) followed by "Verdetto: <symbol>"
    pattern = re.compile(
        r"REQ[-\s]?(\d{1,2})[^\n]*\n.*?Verdetto[:\s*_]*\s*(✅|⚠️|❌)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        req_num = int(m.group(1))
        if 1 <= req_num <= len(_ENFSI_CHECKLIST):
            verdicts[req_num] = m.group(2)
    return verdicts


def _build_deterministic_summary(text: str) -> str:
    """
    Parse verdicts from LLM output and build SCANSIONE VERDETTI + riepilogo
    deterministically in Python — counts are always correct regardless of model output.
    """
    n = len(_ENFSI_CHECKLIST)
    verdicts = _parse_verdicts(text)

    # Need at least half the verdicts to produce a meaningful summary
    if len(verdicts) < n // 2:
        return ""

    conformi = sorted(k for k, v in verdicts.items() if v == "✅")
    parziali  = sorted(k for k, v in verdicts.items() if v == "⚠️")
    mancanti  = sorted(k for k, v in verdicts.items() if v == "❌")
    n_conf, n_parz, n_manc = len(conformi), len(parziali), len(mancanti)

    def fmt(nums: list[int]) -> str:
        return ", ".join(f"REQ-{r:02d}" for r in nums) if nums else "nessuno"

    # SCANSIONE VERDETTI grid — 5 per row
    grid_rows = []
    for i in range(0, n, 5):
        row = "  ".join(
            f"REQ-{r:02d}: {verdicts.get(r, '❓')}"
            for r in range(i + 1, min(i + 6, n + 1))
        )
        grid_rows.append(row)
    grid = "\n".join(grid_rows)

    if n_conf >= 15:
        judgment = "Buona conformità — lacune minori facilmente integrabili"
    elif n_conf >= 10:
        judgment = "Conformità discreta — alcune integrazioni necessarie prima della presentazione"
    elif n_conf >= 5:
        judgment = "Conformità parziale — interventi sostanziali necessari prima della presentazione in giudizio"
    else:
        judgment = "Non conforme — revisione strutturale richiesta"

    return f"""

---

## Scansione Verdetti

{grid}

---

## Riepilogo conformità ENFSI BPM

✅ Conformi:  {n_conf}/{n} — {fmt(conformi)}
⚠️ Parziali: {n_parz}/{n} — {fmt(parziali)}
❌ Mancanti:  {n_manc}/{n} — {fmt(mancanti)}
**Totale: {n_conf} + {n_parz} + {n_manc} = {n_conf + n_parz + n_manc}**

**Giudizio complessivo: {judgment}**"""


def compliance_check_stream(perizia_text: str) -> Generator[str, None, None]:
    """
    Stream the ENFSI compliance analysis of a perizia.
    Uses Ollama /api/generate with a dedicated system prompt (role anchor) and a
    separate user prompt (perizia text only), plus an enlarged context window.
    Yields the full accumulated text at each step (same contract as rag_chat_stream).
    """
    from core.rag import OLLAMA_URL, _rag_model

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(perizia_text)

    accumulated = ""
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": _rag_model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "num_ctx": 16384,  # fits system prompt + 24k perizia + full 20-block output
                "temperature": 0.1,
                "repeat_penalty": 1.1,
            },
        },
        stream=True,
        timeout=300,
    ) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done"):
                    accumulated += data.get("response", "")
                    yield accumulated

    # Append deterministic summary built from parsed verdicts — never wrong
    summary = _build_deterministic_summary(accumulated)
    if summary:
        yield accumulated + summary
