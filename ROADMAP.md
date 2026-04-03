# GraphoLab — Commercial Roadmap

> This document outlines the development roadmap for evolving GraphoLab from a demo laboratory into a commercial product.

---

## From Labs to Product

| GraphoLab Labs (current) | GraphoLab Commercial (future) |
|---|---|
| Jupyter Notebooks | Professional web application |
| Gradio multi-tab demo | Case management dashboard |
| No data persistence | Database of cases, documents, reports |
| No authentication | User auth, roles, audit log |
| No reporting | Automatic PDF report generation |
| Single-file processing | Batch processing on document archives |

---

## Architecture Decisions (confirmed 2026-03-31)

| Decision | Choice | Rationale |
|---|---|---|
| Frontend | **React + shadcn/ui** | Largest ecosystem, professional components ready-to-use |
| Backend API | **FastAPI** | Async, automatic OpenAPI docs, Python-native |
| Database | **PostgreSQL** | Cases, users, audit log |
| File storage | **MinIO** (S3-compatible) | Documents and images, on-premise |
| Auth | **JWT + bcrypt** | Keycloak/SSO deferred to enterprise phase |
| PDF reports | **WeasyPrint** or **ReportLab** | Forensic report generation |
| Deployment | **On-premise Docker** | Primary target; forensic data must not leave client network |
| AI logic | **`core/` shared package** | Reused by both Gradio demo and FastAPI backend |

### Key architectural principles

#### 1. Thin frontend (dumb client)

The React frontend does zero processing. It only renders UI and makes HTTP calls to the backend.
All logic — AI, validation, business rules, auth — lives in the backend or in `core/`.

#### 2. Shared `core/` package

All AI/ML logic lives in pure Python modules under `core/`, with no dependency on any web framework.
`core/` is called by the FastAPI backend (for the professional app) and directly by `grapholab_demo.py` (for the Gradio demo).

#### 3. Every feature follows the same three-layer pattern

- `core/<module>.py` — pure AI/business logic, no HTTP, fully testable in isolation
- `backend/routers/<module>.py` — FastAPI router: receives request, calls `core/`, returns JSON
- `frontend/src/...` — React component: calls the endpoint, renders the result

#### 4. Gradio demo preserved

`app/grapholab_demo.py` is preserved as-is for demos and HF Spaces.
It calls `core/` directly — acceptable because it is a local single-user demo, not a multi-user web app.

```text
grapholab/
├── core/                    # shared AI logic (new)
│   ├── ocr.py
│   ├── signature.py
│   ├── graphology.py
│   ├── ner.py
│   ├── writer.py
│   ├── pipeline.py
│   ├── dating.py
│   └── rag.py
├── app/
│   └── grapholab_demo.py    # Gradio demo (preserved, refactored to import from core/)
├── backend/                 # FastAPI professional app (new)
└── frontend/                # React + shadcn/ui SPA (new)
```

---

## Priority Issue: Dependency Licenses

**Two dependencies carry AGPL-3.0 licenses**, which impose obligations on any commercial product.

| Dependency | License | Commercial impact |
|---|---|---|
| ~~`ultralytics` (YOLOv8)~~ ✅ | ~~AGPL-3.0~~ | Replaced with `transformers` + Conditional DETR (Apache 2.0) |
| `albumentations` | AGPL-3.0 / Commercial | Same constraint |
| All others | BSD / Apache 2.0 / MIT | No commercial restrictions |

**Resolution (recommended before any commercial release):**

- ~~Replace `ultralytics` with RT-DETR via `transformers` (Apache 2.0) — covers the Lab 04 signature detection use case~~ ✅ Done (Conditional DETR)
- ~~Remove `albumentations` from requirements (not used in production code, only in notebooks)~~ ✅ Done

---

## Target Market

GraphoLab Commercial is a **software platform for the forensic analysis of handwritten documents and signatures**, targeting:

- Law firms and notarial offices
- Forensic document examiners and graphologists
- Courts and law enforcement agencies
- Banks and financial institutions (cheque and contract anti-fraud)
- Historical archives and libraries

---

## Deployment Strategy

### Why on-premise is the natural fit for forensic work

Forensic data (signatures, manuscripts, legal documents) is highly sensitive. In professional practice:

- Examiners strongly prefer to process data **locally**, not send it to external servers
- **Chain of custody** requirements mandate full traceability
- **GDPR** (EU) requires control over data residency

### Deployment options

**Option 1 — On-premise Docker** *(primary, confirmed)*
- Extend the existing `docker-compose.yml`
- Client installs on a local server or company VM
- Multi-user, browser access from internal network
- Easy to update; no data leaves the network

**Option 2 — Standalone Desktop** *(for individual practitioners, future)*
- Windows/macOS installer (.exe / .dmg)
- FastAPI backend bundled with PyInstaller + Electron UI
- Fully local, works offline, no network dependencies

**Option 3 — SaaS Cloud** *(optional, future)*
- Hosted on AWS / Azure / GCP
- More complex GDPR compliance; expect resistance from forensic clients
- Secondary channel once on-premise is established

---

## Development Roadmap

### Phase 0 — Core Module Extraction (1–2 weeks)

Branch: `feature/core-modules`

Extract AI logic from `grapholab_demo.py` into a shared `core/` package.
`grapholab_demo.py` remains fully functional throughout — it becomes a thin Gradio wrapper.

Migration order (most independent first):

- [x] Create `core/__init__.py`
- [x] `core/ner.py` — NER pipeline
- [x] `core/ocr.py` — TrOCR + EasyOCR
- [x] `core/graphology.py` — HOG, LBP, graphological analysis
- [x] `core/writer.py` — writer identification
- [x] `core/signature.py` — SigNet + Conditional DETR
- [x] `core/rag.py` — RAG + Ollama
- [x] `core/dating.py` — document dating (uses OCR internally)
- [x] `core/pipeline.py` — full forensic pipeline (aggregates all others)
- [x] Update `grapholab_demo.py` to import from `core/`
- [x] Verify Gradio demo works identically
- [x] Replace `ultralytics` with Conditional DETR in `core/signature.py`
- [x] Remove `albumentations` from `requirements.txt`

### Phase 1 — MVP (6–8 weeks)

Branch: `feature/backend` (from `feature/core-modules`)

Professional core that replaces the Jupyter notebooks.

- [x] FastAPI skeleton + PostgreSQL schema (`users`, `organizations`, `projects`, `documents`, `analyses`, `audit_log`)
- [x] JWT authentication (login, logout, refresh, password reset)
- [x] Role-based access: admin, examiner, viewer
- [x] MinIO integration for document/image storage
- [x] CRUD projects per user
- [x] 4 AI engines via REST API (HTR, signature verification, signature detection, graphological analysis)
- [x] Immutable audit log (forensic requirement — append-only, no UPDATE/DELETE)
- [x] PDF report generation (ReportLab — with images and formatted tables)
- [x] Docker Compose updated with: `postgres`, `minio`, `backend`
- [x] RAG / Consulente Forense IA (Ollama + nomic-embed-text)

Branch: `feature/frontend` (from `feature/backend`)

- [x] React + Tailwind CSS + shadcn/ui scaffold
- [x] Auth pages (login, logout, password reset)
- [x] Case management dashboard (list, create, delete projects)
- [x] Analysis UI for 4 core engines
- [ ] Report download
- [x] Docker Compose updated with: `frontend`

### Phase 2 — Mature Product (6–8 weeks)

- [ ] Report download dal frontend
- [ ] Multilingual support (react-i18next): Italian + English (struttura i18n già presente)
- [ ] Batch processing (ZIP archive → automatic pipeline → aggregate report)
- [ ] Side-by-side sample comparison with annotations (Fabric.js or Konva.js)
- [ ] Full-text search in archived OCR content (PostgreSQL FTS or Meilisearch)
- [ ] Data export (CSV, JSON) for third-party system integration
- [ ] AI model updates without reinstallation
- [ ] End-user documentation and examiner manual
- [x] **Perizia Compliance Checker**: upload PDF perizia esistente → analisi conformità linee guida ENFSI (checklist codificata nel prompt) → report strutturato con ✅ conformità / ⚠️ parziali / ❌ mancanti + raccomandazioni numerate (Ollama LLM, streaming SSE)
- [ ] **Verifica Firma Grafometrica**: upload due file CSV/XLSX con dati grafometrici (pressione, velocità, coordinate XY) → DTW + cosine similarity su feature aggregate → verdict con report comparativo e score 0-100

### Phase 3 — Enterprise (variable)

- [ ] SSO / LDAP / Active Directory integration (Keycloak)
- [ ] Multi-tenancy (multiple organisations on the same instance)
- [ ] Public REST API with OpenAPI docs (FastAPI auto-generates this)
- [ ] Fine-tuning interface for client-proprietary datasets
- [ ] Usage analytics dashboard for admins
- [ ] SLA support agreements
- [ ] Grafometrics avanzato: rendering tracciato XY → immagine → SigNet per confronto visuale integrato con firma image-based

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| AI / ML | PyTorch, Transformers, OpenCV | Unchanged from labs |
| Shared AI package | `core/` (new) | Reused by Gradio demo and FastAPI |
| Backend API | FastAPI | Async, automatic OpenAPI docs |
| Frontend | React + Tailwind CSS + shadcn/ui | Professional components |
| Database | PostgreSQL | Cases, users, audit log |
| File storage | MinIO (S3-compatible) | Documents and images, on-premise |
| Auth | JWT + bcrypt | Keycloak for enterprise SSO (Phase 3) |
| PDF reports | WeasyPrint or ReportLab | Forensic report generation |
| i18n | react-i18next | Italian + English (Phase 2) |
| Annotations | Fabric.js or Konva.js | Interactive image annotation (Phase 2) |
| Container | Docker Compose | Extend existing setup |
| CI/CD | GitHub Actions | Build, test, automated releases |

---

## Licensing Model

### Recommended: Open-Core

| Edition | License | Target |
|---|---|---|
| **Community** | Apache 2.0, open source | Individual researchers, academia |
| **Professional** | Commercial, closed source | Examiners, law firms |
| **Enterprise** | Commercial + SLA contract | Courts, banks, government |

### Customer pricing (indicative)

| Plan | Target | Model |
|---|---|---|
| **Solo** | Single examiner | Annual licence, 1 installation |
| **Studio** | 2–10 users | Annual licence per user |
| **Enterprise** | Courts, banks | Contract + SLA + fine-tuning |
| **SaaS** *(future)* | Small firms | Monthly subscription / pay-per-use |
