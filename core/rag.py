"""
GraphoLab core — RAG (Retrieval-Augmented Generation) + Ollama integration.

Provides:
  - check_ollama()          check whether Ollama server is reachable
  - ollama_list_models()    list available models
  - set_rag_model()         change the active generation model
  - rag_load_docs()         load synthetic + cached documents at startup
  - rag_add_docs()          index new uploaded PDF/DOCX files
  - rag_remove_doc()        remove a document from the index
  - rag_doc_list()          list indexed documents
  - rag_doc_choices()       list indexed document names
  - rag_retrieve()          retrieve top-k chunks for a query
  - stream_ollama()         stream tokens from Ollama /api/generate
  - rag_chat_stream()       full RAG chat: retrieve + build prompt + stream tokens
  - pipeline_llm_synthesis() LLM synthesis of forensic pipeline results
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Generator

import numpy as np
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

_embed_model = OLLAMA_MODEL   # embedding model — changing it invalidates cache
_rag_model = OLLAMA_MODEL     # generation model — selectable via UI

# ──────────────────────────────────────────────────────────────────────────────
# In-memory state
# ──────────────────────────────────────────────────────────────────────────────

_rag_chunks: list = []
_rag_indexed_files: set = set()
_rag_ready = False
_rag_lock = threading.Lock()

# Built-in synthetic knowledge base
_RAG_SYNTHETIC_DOCS = [
    (
        "Analisi della pressione",
        "La pressione grafica indica la forza con cui la penna o la matita viene premuta sul foglio. "
        "Una pressione forte (tratti profondi, rilevabili anche sul retro del foglio) è associata a "
        "carattere deciso, vitalità e a volte aggressività. Una pressione leggera (tratti quasi "
        "impercettibili) può indicare sensibilità, adattabilità o, in contesti patologici, stanchezza "
        "e astenia. La pressione irregolare — alternanza di tratti forti e deboli nello stesso scritto — "
        "può segnalare instabilità emotiva, stati di ansia o condizioni neurologiche. In grafologia "
        "forense la pressione è fondamentale per distinguere scritture apposte in condizioni normali "
        "da quelle prodotte sotto costrizione fisica o psicologica.",
    ),
    (
        "Inclinazione del tratto",
        "L'inclinazione della scrittura descrive l'angolo dei tratti verticali delle lettere rispetto "
        "alla riga di base. Una scrittura verticale (0°) indica equilibrio e obiettività. "
        "L'inclinazione a destra (>15°) è associata a estroversia, impulsività e orientamento verso "
        "il futuro. L'inclinazione a sinistra (<−10°) può indicare introversione, tendenza al ripiegamento "
        "su se stessi o, in contesti forensi, un tentativo di camuffare la propria calligrafia. "
        "L'inclinazione variabile (misto destra/sinistra nello stesso testo) è indicatore di "
        "instabilità emotiva. La misurazione forense dell'inclinazione avviene tramite analisi "
        "angolare dei tratti ascendenti (h, l, b, f) e discendenti (g, p, q).",
    ),
    (
        "Spaziatura grafica",
        "La spaziatura riguarda la distanza tra lettere, parole e righe. Spaziatura ampia tra le parole "
        "indica bisogno di spazio personale, pensiero indipendente e, talvolta, solitudine. "
        "Spaziatura ridotta (parole quasi attaccate) è correlata a socievolezza eccessiva, difficoltà "
        "nei confini relazionali e, in casi estremi, pensiero confusionario. La spaziatura irregolare — "
        "alternanza di parole distanti e ravvicinate — è un indicatore di disorganizzazione cognitiva "
        "o di scrittura non spontanea (es. copiatura o dettatura lenta). In perizie forensi, "
        "la spaziatura viene misurata in millimetri su campioni standardizzati.",
    ),
    (
        "Margini e layout",
        "I margini del foglio riflettono il rapporto dello scrittore con l'ambiente e il contesto "
        "sociale. Un margine sinistro ampio e costante indica rispetto delle regole e pianificazione. "
        "Un margine sinistro che si allarga progressivamente (testo che 'scivola' verso destra) "
        "suggerisce entusiasmo crescente o impulsività. Margine destro ampio è associato a prudenza, "
        "timore del futuro e riservatezza. L'assenza di margini (testo che occupa tutto il foglio) "
        "indica esuberanza comunicativa o senso di urgenza. In perizia, il margine aiuta a "
        "distinguere scritti autentici da trascrizioni o copie, poiché l'autore mantiene "
        "inconsciamente le proprie abitudini spaziali.",
    ),
    (
        "Firme autentiche",
        "Una firma autentica possiede caratteristiche di naturalezza e fluidità del movimento. "
        "I tratti sono continui, con accelerazione e decelerazione tipiche del gesto automatizzato. "
        "La pressione varia in modo coerente con il ritmo del tratto. I legamenti tra le lettere "
        "sono coerenti con il corpus grafico dello scrittore. La firma autentica presenta micro-tremori "
        "naturali (diversi dai tremori patologici) e piccole variazioni tra esecuzioni successive, "
        "mai perfettamente identiche. In perizia calligrafica, si confrontano almeno 10-15 firme "
        "autentiche per stabilire la 'gamma di variazione naturale' prima di esaminare la firma contestata.",
    ),
    (
        "Firme false",
        "Le firme contraffatte si distinguono per diversi indicatori: velocità di esecuzione "
        "innaturalmente lenta (visibile nei 'tocchi' del pennino e nelle esitazioni), tremori "
        "artificiali (regolari, non spontanei), ritocchi e correzioni del tratto, interruzioni "
        "anomale del gesto. La falsificazione per imitazione diretta (calco o copia visiva) produce "
        "una firma con aspetto simile all'originale ma con movimenti invertiti rispetto alla direzione "
        "naturale. Il falsario tende a concentrarsi sulla forma complessiva trascurando i dettagli "
        "minuti (proporzioni tra lettere, angolo di attacco del tratto, pressione). "
        "L'analisi forense utilizza ingrandimenti 10x-40x e, nei casi dubbi, grafometria digitale.",
    ),
    (
        "Velocità e ritmo",
        "La velocità di scrittura si manifesta nella forma delle lettere (semplificazione dei tratti "
        "in scrittura rapida), nell'inclinazione (più marcata ad alta velocità), nelle legature "
        "(frequenti in scrittura veloce, assenti in quella lenta). Il ritmo è la regolarità con cui "
        "si alternano tensione e distensione nel movimento grafico. Un ritmo regolare indica "
        "equilibrio psicofisico. Un ritmo aritmico (alternanza caotica di tratti tesi e distesi) "
        "può segnalare stati emotivi alterati, patologie neurologiche o scrittura non spontanea. "
        "In perizia forense la velocità è cruciale: una firma depositata 'lentamente' da una persona "
        "abitualmente veloce è un forte indicatore di contraffazione.",
    ),
    (
        "Datazione documenti",
        "La datazione grafica di un documento si basa su elementi intrinseci ed estrinseci. "
        "Elementi intrinseci: evoluzione dello stile grafico dell'autore nel tempo (campioni noti "
        "datati permettono di costruire una 'curva di evoluzione'), deterioramento della calligrafia "
        "legato all'età, variazioni nelle abitudini punteggiatura e abbreviazioni. "
        "Elementi estrinseci: tipo di inchiostro (analisi spettroscopica), supporto cartaceo "
        "(filigrana, composizione chimica), strumento di scrittura (biro, stilografica, matita). "
        "L'analisi dell'inchiostro mediante cromatografia liquida può stabilire se l'inchiostro "
        "è compatibile con la data dichiarata. In perizia, la datazione grafica va sempre "
        "abbinata ad analisi chimiche per raggiungere un grado di certezza forense.",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    """Return True if Ollama server is reachable."""
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def ollama_list_models() -> list[str]:
    """Return sorted list of model names available in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return sorted(models) if models else [OLLAMA_MODEL]
    except Exception:
        return [OLLAMA_MODEL]


def set_rag_model(model_name: str) -> str:
    """Set the active Ollama generation model. Returns a status message."""
    global _rag_model
    if model_name:
        _rag_model = model_name
    return f"✅ Modello attivo: **{_rag_model}**"


def stream_ollama(prompt: str) -> Generator[str, None, None]:
    """Yield response tokens from Ollama one at a time (streaming)."""
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": _rag_model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=120,
    ) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done"):
                    yield data.get("response", "")


def _ollama_embed(text: str) -> np.ndarray | None:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": _embed_model, "prompt": text},
            timeout=30,
        )
        return np.array(r.json()["embedding"], dtype=np.float32)
    except Exception:
        return None


def _ollama_embed_batch(texts: list[str]) -> list[np.ndarray | None]:
    """Embed a list of texts. Falls back to sequential calls if batch endpoint unavailable."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": _embed_model, "input": texts},
            timeout=max(30, len(texts) * 3),
        )
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings") or data.get("embedding")
        if embeddings and len(embeddings) == len(texts):
            return [np.array(e, dtype=np.float32) for e in embeddings]
    except Exception:
        pass
    return [_ollama_embed(t) for t in texts]


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rag_cache_path(cache_dir: Path, filename: str, file_bytes: bytes) -> Path:
    h = hashlib.sha256(file_bytes).hexdigest()[:8]
    stem = Path(filename).stem[:40]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return cache_dir / f"{safe}_{h}.npz"


def _rag_cache_save(cache_path: Path, chunks: list, filename: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    good = [c for c in chunks if c["emb"] is not None]
    if not good:
        return
    np.savez_compressed(
        str(cache_path),
        texts=np.array([c["text"] for c in good], dtype=object),
        sources=np.array([c["source"] for c in good], dtype=object),
        embs=np.stack([c["emb"] for c in good]),
        filename=np.array(filename, dtype=object),
    )


def _rag_cache_load(cache_path: Path) -> tuple[list, str]:
    """Returns (chunks, original_filename)."""
    data = np.load(str(cache_path), allow_pickle=True)
    filename = str(data["filename"])
    chunks = [
        {"text": str(t), "source": str(s), "emb": e}
        for t, s, e in zip(data["texts"], data["sources"], data["embs"])
    ]
    return chunks, filename


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str, size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source, "emb": None})
        start += size - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Document index queries
# ──────────────────────────────────────────────────────────────────────────────

def rag_doc_list() -> list[list]:
    """Return rows [[filename, chunk_count]] (user docs only, not synthetic)."""
    synthetic_sources = {s for s, _ in _RAG_SYNTHETIC_DOCS}
    counts: dict = {}
    for c in _rag_chunks:
        src = c["source"]
        if src not in synthetic_sources:
            counts[src] = counts.get(src, 0) + 1
    return [[name, cnt] for name, cnt in sorted(counts.items())]


def rag_doc_choices() -> list[str]:
    return [row[0] for row in rag_doc_list()]


# ──────────────────────────────────────────────────────────────────────────────
# Document loading and indexing
# ──────────────────────────────────────────────────────────────────────────────

def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF, falling back to EasyOCR for scanned pages."""
    full_text = []
    try:
        import pypdf
    except ImportError:
        print(f"[RAG] pypdf not installed — skipping {path.name}")
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if len(page_text.strip()) >= 50:
                full_text.append(page_text)
            else:
                try:
                    import fitz
                    import numpy as np
                    from core.ocr import get_easyocr
                    doc = fitz.open(str(path))
                    fitz_page = doc[page_num]
                    mat = fitz.Matrix(150 / 72, 150 / 72)
                    pix = fitz_page.get_pixmap(matrix=mat)
                    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 4:
                        img_arr = img_arr[:, :, :3]
                    ocr_result = get_easyocr().readtext(img_arr, detail=0, paragraph=True)
                    full_text.append(" ".join(ocr_result))
                    doc.close()
                except ImportError:
                    print(f"[RAG] pymupdf not installed — cannot OCR scanned page {page_num+1}")
                except Exception as e:
                    print(f"[RAG] OCR error on page {page_num+1} of {path.name}: {e}")
    except Exception as e:
        print(f"[RAG] Error reading PDF {path.name}: {e}")
    return "\n".join(full_text)


def rag_load_docs(cache_dir: Path) -> None:
    """Load synthetic knowledge + cached user documents at startup (call once in background)."""
    global _rag_chunks, _rag_indexed_files, _rag_ready
    with _rag_lock:
        chunks: list = []
        for source, text in _RAG_SYNTHETIC_DOCS:
            chunks.extend(_chunk_text(text, source))

        cache_dir.mkdir(parents=True, exist_ok=True)
        for cache_file in sorted(cache_dir.glob("*.npz")):
            try:
                cached_chunks, orig_filename = _rag_cache_load(cache_file)
                chunks.extend(cached_chunks)
                _rag_indexed_files.add(orig_filename)
                print(f"[RAG] Loaded from cache: {orig_filename} ({len(cached_chunks)} chunks)")
            except Exception as e:
                print(f"[RAG] Corrupt cache file {cache_file.name}: {e} — skipping")

        _rag_chunks = chunks
        _rag_ready = True
        print(f"[RAG] Chunks loaded: {len(chunks)} (synthetic + cached)")

    to_embed = [c for c in _rag_chunks if c["emb"] is None]
    if to_embed:
        embeddings = _ollama_embed_batch([c["text"] for c in to_embed])
        embedded = 0
        for chunk, emb in zip(to_embed, embeddings):
            if emb is not None:
                chunk["emb"] = emb
                embedded += 1
        print(f"[RAG] Synthetic embedding done: {embedded} chunks")


def rag_add_docs(files: list, cache_dir: Path) -> tuple[str, list]:
    """Index uploaded PDF/DOCX files. Returns (status_message, doc_list)."""
    global _rag_indexed_files
    if not files:
        return "Nessun file caricato.", rag_doc_list()
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except Exception:
        return (
            "❌ Ollama non raggiungibile — i documenti non possono essere indicizzati.\n"
            "Avvia `ollama serve` e ricarica.",
            rag_doc_list(),
        )
    lines = []
    for f in files:
        path = Path(f.name)
        suffix = path.suffix.lower()
        if path.name in _rag_indexed_files:
            lines.append(f"ℹ️ `{path.name}` — già indicizzato, saltato.")
            continue

        file_bytes = path.read_bytes()
        cache_path = _rag_cache_path(cache_dir, path.name, file_bytes)

        if cache_path.exists():
            try:
                cached_chunks, _ = _rag_cache_load(cache_path)
                with _rag_lock:
                    _rag_chunks.extend(cached_chunks)
                    _rag_indexed_files.add(path.name)
                lines.append(f"✅ `{path.name}` — {len(cached_chunks)} chunk caricati dalla cache.")
                continue
            except Exception:
                pass

        try:
            if suffix == ".pdf":
                text = _extract_pdf_text(path)
            elif suffix in (".docx", ".doc"):
                import docx as _docx
                doc_obj = _docx.Document(str(path))
                text = "\n".join(p.text for p in doc_obj.paragraphs)
            else:
                lines.append(f"⚠️ `{path.name}` — formato non supportato (solo PDF/DOCX).")
                continue
        except Exception as e:
            lines.append(f"❌ `{path.name}` — errore: {e}")
            continue

        if not text.strip():
            lines.append(f"⚠️ `{path.name}` — nessun testo estratto.")
            continue

        chunks = _chunk_text(text, path.name)
        embeddings = _ollama_embed_batch([c["text"] for c in chunks])
        embedded = 0
        for chunk, emb in zip(chunks, embeddings):
            if emb is not None:
                chunk["emb"] = emb
                embedded += 1

        try:
            _rag_cache_save(cache_path, chunks, path.name)
        except Exception as e:
            print(f"[RAG] Cache write failed for {path.name}: {e}")

        with _rag_lock:
            _rag_chunks.extend(chunks)
            _rag_indexed_files.add(path.name)
        lines.append(f"✅ `{path.name}` — {len(chunks)} chunk, {embedded} indicizzati.")

    return "\n".join(lines), rag_doc_list()


def rag_remove_doc(filename: str, cache_dir: Path) -> tuple[str, list]:
    """Remove all chunks for a document from memory and delete its cache file."""
    global _rag_chunks, _rag_indexed_files
    if not filename or not filename.strip():
        return "Nessun documento selezionato.", rag_doc_list()

    with _rag_lock:
        before = len(_rag_chunks)
        _rag_chunks = [c for c in _rag_chunks if c["source"] != filename]
        removed_chunks = before - len(_rag_chunks)
        _rag_indexed_files.discard(filename)

    deleted_files = 0
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.npz"):
            try:
                with np.load(str(cache_file), allow_pickle=True) as data:
                    match = str(data["filename"]) == filename
                if match:
                    cache_file.unlink()
                    deleted_files += 1
            except Exception:
                pass

    if removed_chunks == 0:
        return f"⚠️ `{filename}` non trovato nell'indice.", rag_doc_list()

    msg = f"🗑️ `{filename}` rimosso ({removed_chunks} chunk eliminati"
    if deleted_files:
        msg += ", cache eliminata"
    msg += ")."
    return msg, rag_doc_list()


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def rag_retrieve(question: str) -> tuple[list | None, str | None]:
    """Return (results, error_str). results is list of (score, chunk)."""
    embedded_chunks = [c for c in _rag_chunks if c["emb"] is not None]
    if not embedded_chunks:
        total = len(_rag_chunks)
        return None, (
            f"⏳ Embedding in corso (0/{total} chunk pronti). "
            "Riprovare tra qualche secondo — l'indicizzazione procede in background."
        )
    q_emb = _ollama_embed(question)
    if q_emb is None:
        return None, "❌ Impossibile generare l'embedding della domanda. Ollama è in esecuzione?"

    synthetic_sources = {s for s, _ in _RAG_SYNTHETIC_DOCS}
    user_chunks = [c for c in _rag_chunks if c["emb"] is not None and c["source"] not in synthetic_sources]
    synth_chunks = [c for c in _rag_chunks if c["emb"] is not None and c["source"] in synthetic_sources]

    def _top_k_from(pool, q, k):
        if not pool:
            return []
        embs = np.stack([c["emb"] for c in pool])
        q_n = q / (np.linalg.norm(q) + 1e-9)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        scores = (embs / norms) @ q_n
        idxs = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), pool[i]) for i in idxs]

    user_results = _top_k_from(user_chunks, q_emb, 2)
    synth_results = _top_k_from(synth_chunks, q_emb, 2 if user_results else 4)
    return user_results + synth_results, None


# ──────────────────────────────────────────────────────────────────────────────
# Chat stream (framework-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def rag_chat_stream(
    message: str,
    history: list[dict],
) -> Generator[tuple[str, str | None], None, None]:
    """Core RAG chat logic. Yields (partial_response, sources_footer|None).

    The caller (e.g. Gradio wrapper) is responsible for formatting history.
    history is a list of {"role": "user"|"assistant", "content": str}.
    """
    if not check_ollama():
        yield (
            "❌ **Ollama non raggiungibile.**\n\n"
            "Avvia il server con:\n```\nollama serve\n```\n"
            "e assicurati che il modello sia scaricato:\n"
            "```\nollama pull llama3.2\n```",
            None,
        )
        return

    if not _rag_ready:
        yield "⏳ Indice della knowledge base in costruzione, riprovare tra qualche secondo…", None
        return

    results, err = rag_retrieve(message)
    if err:
        yield err, None
        return

    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for _, c in results)

    recent = history[-12:] if len(history) > 12 else history
    conv_text = ""
    i = 0
    while i < len(recent) - 1:
        if recent[i]["role"] == "user" and recent[i + 1]["role"] == "assistant":
            u = recent[i]["content"]
            a = recent[i + 1]["content"].split("\n\n---\n")[0]
            conv_text += f"Utente: {u}\nAssistente: {a}\n\n"
            i += 2
        else:
            i += 1

    prompt = (
        "Sei un esperto di grafologia forense. Rispondi in italiano, in modo preciso e "
        "conciso, basandoti ESCLUSIVAMENTE sui seguenti estratti.\n\n"
        f"{context}\n\n"
    )
    if conv_text:
        prompt += f"Conversazione precedente:\n{conv_text}\n"
    prompt += f"Domanda: {message}\n\nRisposta:"

    sources = list(dict.fromkeys(c["source"] for _, c in results))
    sources_footer = f"\n\n---\n*Fonti: {', '.join(sources)}*"

    partial = ""
    try:
        for token in stream_ollama(prompt):
            partial += token
            yield partial, None
    except Exception as e:
        yield f"❌ Errore nella generazione: {e}", None
        return

    yield partial, sources_footer


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline LLM synthesis
# ──────────────────────────────────────────────────────────────────────────────

def pipeline_llm_synthesis(
    step1_summary: str,
    step2_text: str,
    step3_summary: str,
    step4_report: str,
    step5_report: str,
    step6_report: str,
) -> str:
    """Call Ollama to synthesise forensic pipeline results into a narrative report."""
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except Exception:
        return (
            "❌ **Ollama non raggiungibile.** Avvia il server con:\n"
            "```\nollama serve\n```"
        )
    prompt = (
        "Sei un perito calligrafo forense esperto. "
        "Sulla base delle seguenti analisi tecniche su un documento, "
        "fornisci in italiano una valutazione complessiva professionale: "
        "evidenzia elementi di interesse forense, coerenze e incoerenze tra i risultati, "
        "e suggerisci eventuali ulteriori verifiche.\n\n"
        f"=== RILEVAMENTO FIRMA ===\n{step1_summary}\n\n"
        f"=== TRASCRIZIONE HTR ===\n{step2_text}\n\n"
        f"=== ENTITÀ RICONOSCIUTE (NER) ===\n{step3_summary}\n\n"
        f"=== IDENTIFICAZIONE AUTORE ===\n{step4_report}\n\n"
        f"=== ANALISI GRAFOLOGICA ===\n{step5_report}\n\n"
        f"=== VERIFICA FIRMA ===\n{step6_report}\n\n"
        "Valutazione forense integrata:"
    )
    result = ""
    try:
        for token in stream_ollama(prompt):
            result += token
    except Exception as e:
        return f"❌ Errore nella generazione LLM: {e}"
    return result if result else "*(Nessuna risposta dal modello)*"
