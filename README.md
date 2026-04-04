---
title: GraphoLab — AI for Forensic Graphology
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.25.0"
app_file: app/grapholab_demo.py
pinned: false
license: apache-2.0
short_description: AI-powered forensic graphology platform (HTR, signatures, NER, ENFSI compliance)
---

# GraphoLab — Forensic Graphology Laboratory

An AI-powered platform for **forensic graphology**: scientific examination of handwriting and signatures for legal purposes.

GraphoLab ships in two forms:

| Mode | Description |
|------|-------------|
| **Professional app** | FastAPI backend + React frontend — multi-user, JWT auth, PostgreSQL, MinIO, audit log |
| **Gradio demo** | Single-user interactive demo, runs locally or on Hugging Face Spaces |

---

## AI Capabilities

| Engine | Technique | Model |
|--------|-----------|-------|
| Handwritten OCR (HTR) | Transformer OCR | `microsoft/trocr-base-handwritten` + EasyOCR |
| Signature Verification | Siamese Network | SigNet (luizgh/sigver) |
| Signature Detection | Object Detection | Conditional DETR (tech4humans, Apache 2.0) |
| Named Entity Recognition | Token Classification | `Babelscape/wikineural-multilingual-ner` |
| Writer Identification | HOG + SVM | scikit-learn |
| Graphological Analysis | Image Processing | OpenCV + scikit-image |
| Document Dating | OCR + dateparser | EasyOCR + multilingual date parsing |
| Full Forensic Pipeline | All engines in sequence | Ollama LLM synthesis |
| RAG / AI Consultant | Retrieval-Augmented Generation | Ollama + nomic-embed-text |
| **ENFSI Compliance Checker** | LLM structured analysis | Ollama (qwen3:8b recommended) |
| **Agente Documentale** | LangChain ReAct agent + tools | Ollama + LangChain + PaddleOCR |

---

## Professional App (FastAPI + React)

### Features

- **Case management**: create, manage and archive forensic cases
- **Document storage**: MinIO S3-compatible storage (on-premise)
- **AI analysis**: all 8 engines via REST API with streaming SSE
- **PDF report generation**: forensic report with images and formatted tables
- **RAG chatbot**: upload PDF/DOCX to build a knowledge base, query with local LLM
- **ENFSI Compliance Checker**: upload a perizia PDF → LLM analysis against 20 ENFSI BPM-FHX-01 Ed.03 requirements → structured report with ✅/⚠️/❌ verdicts, suggestions, PDF export
- **Agente Documentale**: LangChain ReAct agent with document tools (OCR, NER, graphology, dating) — upload files, run tool-augmented multi-turn conversations, stop mid-stream
- **OCR model selector**: switch between EasyOCR, TrOCR, PaddleOCR, VLM from the sidebar at runtime
- **Immutable audit log**: append-only forensic chain of custody
- **JWT authentication**: login, refresh, password reset
- **Role-based access**: admin, examiner, viewer
- **Multilingual UI**: Italian / English (react-i18next)

### Quick Start (Docker)

```bash
# Copy and edit environment variables
cp .env.example .env   # set SECRET_KEY at minimum

# Start all services (PostgreSQL, MinIO, backend, frontend, Ollama)
docker compose up

# Services:
#   Frontend  → http://localhost:3000
#   Backend   → http://localhost:8000/docs  (OpenAPI)
#   MinIO     → http://localhost:9001       (admin console)
#   Ollama    → http://localhost:11434

# Pull a model for LLM features (recommended: qwen3:8b for RTX 4070 8GB)
ollama pull qwen3:8b
```

### Quick Start (local development)

```bash
# Backend
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r requirements-backend.txt
uvicorn backend.main:app --reload

# Frontend
cd frontend
npm install
npm run dev   # http://localhost:5173
```

### Architecture

```
grapholab/
├── core/                    # Shared AI logic (no web framework dependency)
│   ├── ocr.py               # TrOCR + EasyOCR
│   ├── signature.py         # SigNet + Conditional DETR
│   ├── graphology.py        # HOG, LBP, graphological features
│   ├── ner.py               # Named entity recognition
│   ├── writer.py            # Writer identification
│   ├── dating.py            # Document dating
│   ├── pipeline.py          # Full forensic pipeline
│   ├── rag.py               # RAG + Ollama integration
│   ├── compliance.py        # ENFSI compliance checker
│   └── agent.py             # LangChain ReAct agent + document tools
├── backend/                 # FastAPI professional app
│   ├── routers/             # auth, users, projects, analysis, rag, compliance, agent, audit
│   ├── models/              # SQLAlchemy models
│   └── storage/             # MinIO client
├── frontend/                # React + Tailwind CSS + shadcn/ui SPA
│   └── src/
│       ├── pages/           # ProjectsPage, ProjectPage, RagPage, CompliancePage, AgentPage, AdminPage
│       └── components/
├── app/
│   └── grapholab_demo.py    # Gradio demo (preserved, imports from core/)
├── notebooks/               # Jupyter labs (01–08)
├── docker-compose.yml
├── requirements.txt         # Core + Gradio dependencies
└── requirements-backend.txt # FastAPI + PostgreSQL + MinIO dependencies
```

### API

The FastAPI backend auto-generates OpenAPI docs at `http://localhost:8000/docs`.

Main endpoint groups:

| Prefix | Description |
|--------|-------------|
| `/auth` | Login, refresh, password reset |
| `/users` | User management |
| `/projects` | Case CRUD + document upload |
| `/analysis` | Run AI engines, download PDF report |
| `/rag` | RAG chatbot, document indexing, model selection |
| `/compliance` | ENFSI compliance check (SSE stream + PDF export) |
| `/agent` | LangChain document agent (SSE stream, file attachment, stop) |
| `/audit` | Immutable activity log |

---

## Gradio Demo

Interactive single-user demo, also available on [Hugging Face Spaces](https://huggingface.co/spaces/fabioantonini/grapholab).

### Run locally

```bash
pip install -r requirements.txt
python app/grapholab_demo.py
# Open http://localhost:7860
```

### Tabs

| Tab | Name | Description |
|-----|------|-------------|
| 1 | OCR Manoscritto | Handwritten text transcription |
| 2 | Verifica Firma | Signature verification (SigNet) |
| 3 | Rilevamento Firma | Signature detection (Conditional DETR) |
| 4 | Riconoscimento Entità | Named entity recognition |
| 5 | Identificazione Scrittore | Writer identification (HOG + SVM) |
| 6 | Analisi Grafologica | Graphological feature analysis |
| 7 | Perizia Forense Automatica | Full forensic pipeline + LLM synthesis |
| 8 | Datazione Documenti | Document dating |
| 9 | Consulente Forense IA | RAG chatbot (local Ollama LLM) |

---

## Jupyter Notebooks

| Lab | Notebook | AI Technique |
|-----|----------|-------------|
| 01 | Introduction | Conceptual overview |
| 02 | Handwritten OCR | TrOCR |
| 03 | Signature Verification | SigNet (Siamese network) |
| 04 | Signature Detection | Conditional DETR |
| 05 | Writer Identification | HOG + SVM |
| 06 | Graphological Analysis | OpenCV |
| 07 | Named Entity Recognition | Token classification |
| 08 | dots.ocr VLM | Vision-Language Model (1.7B) |

```bash
jupyter lab notebooks/
```

---

## Requirements

- Python 3.11–3.13
- NVIDIA GPU recommended (CUDA 12.x) — CPU fallback available
- [Ollama](https://ollama.com) for LLM features (pipeline synthesis, RAG, compliance checker)
  - Recommended model: `qwen3:8b` (fits in 8GB VRAM, RTX 4070 Laptop GPU)
- Docker + nvidia-container-toolkit for containerized GPU inference
- [LangChain](https://python.langchain.com) + `langchain-ollama` for the Document Agent

---

## Key Models & Resources

| Use case | Resource |
|----------|----------|
| Handwritten OCR | [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) |
| Signature Detection | [tech4humans/conditional-detr-50-signature-detector](https://huggingface.co/tech4humans/conditional-detr-50-signature-detector) (Apache 2.0) |
| Signature Verification | [luizgh/sigver](https://github.com/luizgh/sigver) |
| NER | [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) |
| Embeddings (RAG) | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama |
| LLM inference | [Ollama](https://ollama.com) — local, no data sent online |
| ENFSI standard | BPM-FHX-01 Ed.03 — Best Practice Manual for Forensic Examination of Handwriting |
| Document OCR (agent) | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — layout + text detection |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
