"""
GraphoLab core — Agentic Document Analysis.

Provides:
  - create_forensic_agent()   build a LangChain AgentExecutor with all GraphoLab tools
  - agent_stream()            streaming generator for Gradio / FastAPI
  - SUGGESTED_PROMPTS         pre-formatted prompt templates for the UI
  - AGENT_TOOLS_NAMES         list of tool names for status endpoint

The agent uses qwen3 (via Ollama) by default — qwen3 has excellent tool-calling
support. The system prompt instructs the model to respond always in Italian and
to use file paths injected into the user message as [file: /path/to/file].
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Generator, Any, Optional

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

AGENT_MODEL = "qwen3:4b"

_ROOT = Path(__file__).parent.parent
SIGNET_WEIGHTS = _ROOT / "models" / "signet.pth"
WRITER_SAMPLES_DIR = _ROOT / "data" / "samples"

FORENSIC_SYSTEM_PROMPT = (
    "Sei un consulente forense specializzato in grafologia e analisi documentale.\n"
    "Hai accesso a strumenti specializzati per analizzare documenti, scritture a mano e firme.\n"
    "Rispondi SEMPRE in italiano, in modo professionale e dettagliato.\n"
    "Usa gli strumenti nella sequenza più logica per rispondere alla richiesta dell'utente.\n"
    "Quando l'utente allega dei file, i loro percorsi ti vengono forniti nel messaggio "
    "nel formato [file: /percorso/al/file]. Usali come argomenti degli strumenti.\n"
    "Se la richiesta riguarda sia la trascrizione che altre analisi (NER, date, ecc.), "
    "trascrivi prima il testo e poi usa il testo risultante come input per gli altri strumenti.\n"
    "Al termine di ogni risposta, fornisci un breve riepilogo delle analisi effettuate."
)

SUGGESTED_PROMPTS = [
    {"label": "Trascrivi testo",
     "text": "Trascrivi il testo manoscritto nel documento allegato"},
    {"label": "NER",
     "text": "Estrai tutte le entità nominate (persone, luoghi, organizzazioni) dal testo del documento allegato"},
    {"label": "Rileva firma",
     "text": "Rileva ed estrai le firme presenti nel documento allegato"},
    {"label": "Verifica firma",
     "text": "Verifica se la firma nel documento è autentica rispetto alla firma di riferimento allegata"},
    {"label": "Grafologia",
     "text": "Analizza la scrittura nel documento e fornisci un profilo grafologico dettagliato"},
    {"label": "Chi ha scritto?",
     "text": "Chi ha scritto questo documento? Confronta la scrittura con i campioni nel database"},
    {"label": "Layout documento",
     "text": "Analizza il layout del documento: identifica tabelle, figure e sezioni di testo"},
    {"label": "Datazione",
     "text": "Trascrivi il documento allegato e cerca le date citate nel testo"},
    {"label": "Pipeline completa",
     "text": "Esegui un'analisi forense completa del documento allegato: trascrizione, entità, firma, grafologia e datazione"},
    {"label": "Analisi tabella",
     "text": "Estrai e struttura i dati dalla tabella presente nel documento allegato"},
    {"label": "Analisi figura",
     "text": "Descrivi e interpreta la figura o il grafico presente nel documento allegato"},
    {"label": "Analisi testamento",
     "text": (
         "Analizza questo testamento: estrai il testo, identifica le persone nominate, "
         "rileva la firma e confrontala con la firma di riferimento allegata"
     )},
]

AGENT_TOOLS_NAMES = [
    "trascrivi_documento",
    "estrai_entita",
    "rileva_firma",
    "verifica_firma",
    "analisi_grafologica",
    "identifica_scrittore",
    "data_documento",
    "analisi_layout",
    "estrai_testo_layout",
    "analizza_tabella",
    "analizza_figura",
    "consulta_knowledge_base",
]

# ──────────────────────────────────────────────────────────────────────────────
# Image loading utility
# ──────────────────────────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    """Load an image file to an RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


# ──────────────────────────────────────────────────────────────────────────────
# LangChain Tools
# ──────────────────────────────────────────────────────────────────────────────

from langchain_core.tools import tool  # noqa: E402


@tool
def trascrivi_documento(image_path: str) -> str:
    """Trascrive il testo manoscritto presente in un documento immagine.

    Usa questo strumento quando l'utente vuole leggere o estrarre testo scritto a mano.
    Il modello OCR usato è quello selezionato dall'utente nella sidebar
    (easyocr / vlm / paddleocr / trocr). Di default usa EasyOCR (CPU, veloce).
    Restituisce il testo trascritto che può essere passato ad altri strumenti.

    Args:
        image_path: Percorso assoluto del file immagine del documento.
    """
    try:
        from core.ocr import htr_transcribe
        img = _load_image(image_path)
        result = htr_transcribe(img)
        return f"Testo trascritto:\n\n{result}"
    except Exception as e:
        return f"Errore nella trascrizione: {e}"


@tool
def estrai_entita(testo: str) -> str:
    """Estrae le entità nominate (persone, luoghi, organizzazioni) da un testo.

    Usa questo strumento dopo aver trascritto un documento, o quando hai già il testo.
    Restituisce un elenco strutturato delle entità trovate.

    Args:
        testo: Il testo da cui estrarre le entità.
    """
    try:
        from core.ner import ner_extract
        _entities, report_md = ner_extract(testo)
        if not _entities:
            return "Nessuna entità nominata trovata nel testo."
        return f"Entità trovate:\n\n{report_md}"
    except Exception as e:
        return f"Errore nell'estrazione delle entità: {e}"


@tool
def rileva_firma(image_path: str) -> str:
    """Rileva le firme presenti in un documento immagine tramite object detection.

    Usa questo strumento per trovare la posizione delle firme in un documento.
    Se vuoi poi verificare l'autenticità di una firma, usa `verifica_firma`.

    Args:
        image_path: Percorso assoluto del file immagine del documento.
    """
    try:
        from core.signature import sig_detect
        img = _load_image(image_path)
        _annotated, summary = sig_detect(img, conf_threshold=0.3)
        return f"Risultato rilevamento firme:\n\n{summary}"
    except Exception as e:
        return f"Errore nel rilevamento firma: {e}"


@tool
def verifica_firma(riferimento_path: str, query_path: str) -> str:
    """Verifica l'autenticità di una firma confrontandola con una firma di riferimento.

    Usa questo strumento quando hai due immagini di firme da confrontare:
    una firma autentica nota (riferimento) e una firma da verificare (query).

    Args:
        riferimento_path: Percorso dell'immagine della firma autentica di riferimento.
        query_path: Percorso dell'immagine della firma da verificare.
    """
    try:
        from core.signature import sig_verify
        ref_img = _load_image(riferimento_path)
        query_img = _load_image(query_path)
        report, _chart = sig_verify(ref_img, None, query_img, SIGNET_WEIGHTS)
        return f"Risultato verifica firma:\n\n{report}"
    except Exception as e:
        return f"Errore nella verifica della firma: {e}"


@tool
def analisi_grafologica(image_path: str) -> str:
    """Analizza le caratteristiche grafologiche di una scrittura a mano.

    Estrae metriche quantitative: inclinazione, pressione, dimensioni lettere,
    spaziatura, densità del tratto, componenti connesse.

    Args:
        image_path: Percorso assoluto del file immagine con la scrittura a mano.
    """
    try:
        from core.graphology import grapho_analyse
        img = _load_image(image_path)
        report_md, _annotated = grapho_analyse(img)
        return f"Analisi grafologica:\n\n{report_md}"
    except Exception as e:
        return f"Errore nell'analisi grafologica: {e}"


@tool
def identifica_scrittore(image_path: str) -> str:
    """Identifica chi ha scritto un documento confrontando con i campioni nel database.

    Usa questo strumento per abbinare una scrittura a uno scrittore noto nel sistema.
    Restituisce i candidati ordinati per probabilità.

    Args:
        image_path: Percorso assoluto del file immagine con la scrittura a mano.
    """
    try:
        from core.writer import writer_identify
        img = _load_image(image_path)
        report, _chart = writer_identify(img, WRITER_SAMPLES_DIR)
        return f"Identificazione scrittore:\n\n{report}"
    except Exception as e:
        return f"Errore nell'identificazione dello scrittore: {e}"


@tool
def data_documento(testo: str) -> str:
    """Estrae e normalizza le date presenti nel testo di un documento.

    Usa questo strumento dopo aver trascritto un documento per trovare le date citate.
    Supporta formati italiani e internazionali.

    Args:
        testo: Il testo del documento da cui estrarre le date.
    """
    try:
        from core.dating import extract_dates
        dates = extract_dates(testo)
        if not dates:
            return "Nessuna data trovata nel testo."
        lines = [f"- {raw} → {dt.strftime('%d/%m/%Y')}" for raw, dt in dates]
        return "Date trovate nel documento:\n\n" + "\n".join(lines)
    except Exception as e:
        return f"Errore nell'estrazione delle date: {e}"


@tool
def analisi_layout(image_path: str) -> str:
    """Analizza il layout di un documento identificando regioni strutturate.

    Rileva automaticamente: testo, tabelle, figure, titoli e altri elementi.
    Usa questo strumento per capire la struttura visiva di un documento.

    Args:
        image_path: Percorso assoluto del file immagine del documento.
    """
    try:
        from core.document_layout import detect_layout
        result = detect_layout(image_path)
        if "error" in result:
            return f"Errore layout detection: {result['error']}"
        regions = result.get("regions", [])
        if not regions:
            return "Nessuna regione strutturata rilevata nel documento."
        lines = [
            f"- **{r['label']}** (confidenza: {r.get('score', 0):.0%})"
            for r in regions
        ]
        return f"Layout del documento — {len(regions)} regioni rilevate:\n\n" + "\n".join(lines)
    except Exception as e:
        return f"Errore nell'analisi del layout: {e}"


@tool
def estrai_testo_layout(image_path: str) -> str:
    """Estrae il testo da un documento stampato usando OCR con ordinamento di lettura.

    A differenza di `trascrivi_documento` (ottimizzato per manoscritti), questo
    strumento è pensato per documenti stampati o misti, e mantiene l'ordine di lettura.

    Args:
        image_path: Percorso assoluto del file immagine del documento.
    """
    try:
        from core.document_layout import extract_ordered_text
        text = extract_ordered_text(image_path)
        if not text.strip():
            return "Nessun testo estratto dal documento."
        return f"Testo estratto (ordinato per posizione di lettura):\n\n{text}"
    except Exception as e:
        return f"Errore nell'estrazione del testo con layout: {e}"


@tool
def analizza_tabella(image_path: str, region_index: int = 0) -> str:
    """Estrae e struttura i dati da una tabella presente in un documento.

    Rileva automaticamente la tabella nel documento, la ritaglia e usa un modello
    visuale (qwen3-vl) per estrarne il contenuto in formato Markdown strutturato.
    Ispirato all'approccio del corso DeepLearning.AI Document AI (L6).

    Usa questo strumento quando il documento contiene tabelle di dati da analizzare
    (es. tabelle comparative, dati anagrafici, elenchi strutturati).

    Args:
        image_path:   Percorso assoluto del file immagine del documento.
        region_index: Indice della tabella da analizzare (0 = prima tabella, default).
    """
    try:
        from core.document_layout import analyse_table_region
        result = analyse_table_region(image_path, region_index=region_index, model=AGENT_MODEL)
        if "error" in result:
            return f"Analisi tabella: {result['error']}"
        region = result["region"]
        return (
            f"Tabella #{region_index + 1} estratta "
            f"(posizione: {region['bbox']}, confidenza: {region.get('score', 0):.0%}):\n\n"
            f"{result['markdown']}"
        )
    except Exception as e:
        return f"Errore nell'analisi della tabella: {e}"


@tool
def analizza_figura(image_path: str, region_index: int = 0) -> str:
    """Analizza e descrive una figura o grafico presente in un documento.

    Rileva automaticamente la figura, la ritaglia e usa un modello visuale
    (qwen3-vl) per descriverne il contenuto: tipo di grafico, valori, trend,
    legenda. Ispirato all'approccio del corso DeepLearning.AI Document AI (L6).

    Usa questo strumento quando il documento contiene grafici, diagrammi, immagini
    o figure che necessitano di interpretazione visiva.

    Args:
        image_path:   Percorso assoluto del file immagine del documento.
        region_index: Indice della figura da analizzare (0 = prima figura, default).
    """
    try:
        from core.document_layout import analyse_figure_region
        result = analyse_figure_region(image_path, region_index=region_index, model=AGENT_MODEL)
        if "error" in result:
            return f"Analisi figura: {result['error']}"
        region = result["region"]
        return (
            f"Figura #{region_index + 1} analizzata "
            f"(posizione: {region['bbox']}, confidenza: {region.get('score', 0):.0%}):\n\n"
            f"{result['description']}"
        )
    except Exception as e:
        return f"Errore nell'analisi della figura: {e}"


@tool
def consulta_knowledge_base(domanda: str) -> str:
    """Consulta la knowledge base forense per rispondere a domande teoriche.

    Usa questo strumento per domande su tecniche forensi, standard ENFSI,
    metodi grafologici, interpretazione di parametri, ecc.
    NON usare questo strumento per analizzare file — usa gli altri strumenti per quello.

    Args:
        domanda: La domanda da porre alla knowledge base forense.
    """
    try:
        from core.rag import rag_retrieve, stream_ollama
        results, err = rag_retrieve(domanda)
        if err:
            return f"Errore nella knowledge base: {err}"
        if not results:
            return "Nessun risultato trovato nella knowledge base."
        context = "\n\n".join(chunk["text"] for _score, chunk in results[:3])
        prompt = (
            "Rispondi in italiano alla seguente domanda forense basandoti sul contesto.\n\n"
            f"Contesto:\n{context}\n\n"
            f"Domanda: {domanda}\n\nRisposta:"
        )
        response = "".join(stream_ollama(prompt))
        return response.strip() or "Nessuna risposta generata."
    except Exception as e:
        return f"Errore nella consultazione della knowledge base: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool list
# ──────────────────────────────────────────────────────────────────────────────

_ALL_TOOLS = [
    trascrivi_documento,
    estrai_entita,
    rileva_firma,
    verifica_firma,
    analisi_grafologica,
    identifica_scrittore,
    data_documento,
    analisi_layout,
    estrai_testo_layout,
    analizza_tabella,
    analizza_figura,
    consulta_knowledge_base,
]

# ──────────────────────────────────────────────────────────────────────────────
# Agent factory  (LangGraph — replaces deprecated AgentExecutor)
# ──────────────────────────────────────────────────────────────────────────────

def create_forensic_agent(model: str = AGENT_MODEL) -> Any:
    """Create a LangGraph react agent with all GraphoLab forensic tools.

    Args:
        model: Ollama model name (default: qwen3:8b).

    Returns:
        Compiled LangGraph graph (CompiledGraph).
    """
    import inspect
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model, temperature=0)
    system_msg = SystemMessage(content=FORENSIC_SYSTEM_PROMPT)

    # LangGraph ≥0.2.57 uses `prompt`; older versions use `state_modifier`
    sig = inspect.signature(create_react_agent)
    if "prompt" in sig.parameters:
        return create_react_agent(llm, _ALL_TOOLS, prompt=system_msg)
    else:
        return create_react_agent(llm, _ALL_TOOLS, state_modifier=system_msg)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming helper
# ──────────────────────────────────────────────────────────────────────────────

def get_active_model() -> str:
    """Return the currently active Ollama model.

    Reads _rag_model from core.rag at call time so it always reflects the
    model selected by the user from the sidebar (which calls set_rag_model).
    Falls back to AGENT_MODEL if core.rag is not yet initialised.
    """
    try:
        from core.rag import _rag_model
        return _rag_model or AGENT_MODEL
    except Exception:
        return AGENT_MODEL


def agent_stream(
    message: str,
    file_paths: list[str],
    history: list[dict],
    model: str | None = None,
    stop_event: Optional[threading.Event] = None,
) -> Generator[str, None, None]:
    """Run the forensic agent and stream accumulated response text.

    Each yield replaces the previous one (accumulative SSE pattern, same as RAG chat).

    Args:
        message:    User's text message.
        file_paths: Absolute paths of uploaded files (injected into the message).
        history:    List of {"role": "user"|"assistant", "content": str}.
        model:      Ollama model name. If None, uses the globally active model
                    (set via the sidebar / set_rag_model).

    Yields:
        Accumulated response string (each yield is a complete update, not a delta).
    """
    if model is None:
        model = get_active_model()
    from langchain_core.messages import HumanMessage, AIMessage

    # Copy uploaded files to a short, predictable temp path so the LLM
    # does not mangle long Windows AppData paths when constructing tool args.
    _gl_tmp = Path(tempfile.gettempdir()) / "gl"
    _gl_tmp.mkdir(exist_ok=True)
    short_paths: list[str] = []
    for i, p in enumerate(file_paths):
        ext = Path(p).suffix or ".png"
        dest = _gl_tmp / f"f{i}{ext}"
        shutil.copy2(p, dest)
        short_paths.append(str(dest))

    # Inject file paths into the user message so the agent can use them as tool args
    if short_paths:
        paths_str = "\n".join(f"[file: {p}]" for p in short_paths)
        full_message = f"{message}\n\nFile allegati:\n{paths_str}"
    else:
        full_message = message

    # Build messages list: history + current user message
    messages: list = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=full_message))

    agent = create_forensic_agent(model)

    accumulated = ""
    tool_log: list[str] = []

    try:
        for chunk in agent.stream(
            {"messages": messages},
            stream_mode="updates",
        ):
            if stop_event is not None and stop_event.is_set():
                break

            # ── Agent node: either a tool-call decision or the final answer ──
            if "agent" in chunk:
                for msg in chunk["agent"]["messages"]:
                    tool_calls = getattr(msg, "tool_calls", [])
                    content = getattr(msg, "content", "") or ""
                    if tool_calls:
                        for tc in tool_calls:
                            tool_name = tc.get("name", "tool")
                            tool_args = str(tc.get("args", {}))[:80]
                            entry = f"\n\n🔧 *`{tool_name}({tool_args})`…*"
                            tool_log.append(entry)
                        yield accumulated + "".join(tool_log)
                    elif content:
                        # Final answer from the agent
                        accumulated = content
                        if tool_log:
                            details = (
                                "\n\n<details><summary>__TOOL_LOG__</summary>\n"
                                + "".join(tool_log)
                                + "\n</details>"
                            )
                            yield accumulated + details
                        else:
                            yield accumulated

            # ── Tools node: results returned by tool calls ──
            elif "tools" in chunk:
                for msg in chunk["tools"]["messages"]:
                    content = getattr(msg, "content", "") or ""
                    short = content[:120] + ("…" if len(content) > 120 else "")
                    if tool_log:
                        tool_log[-1] = tool_log[-1].rstrip("…*") + " ✅*"
                    tool_log.append(f"\n> *{short}*")
                    yield accumulated + "".join(tool_log)

    except Exception as e:
        yield f"{accumulated}\n\n❌ Errore dell'agente: {e}"
