# GraphoLab — Commercial Roadmap

> This document outlines considerations and a development roadmap for evolving GraphoLab from a demo laboratory into a commercial product. No implementation is planned at this stage — this is a reference document for future decisions.

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

## Priority Issue: Dependency Licenses

**Two dependencies carry AGPL-3.0 licenses**, which impose obligations on any commercial product — even SaaS deployments must disclose source code unless a commercial license is purchased.

| Dependency | License | Commercial impact |
|---|---|---|
| `ultralytics` (YOLOv8) | AGPL-3.0 | Source disclosure required, or buy Enterprise license |
| `albumentations` | AGPL-3.0 / Commercial | Same constraint |
| All others | BSD / Apache 2.0 / MIT | No commercial restrictions |

**Three options to resolve this before any commercial release:**

**Option A — Replace AGPL dependencies** *(recommended)*
- Replace `ultralytics` with an Apache 2.0-licensed detector (e.g. RT-DETR via `transformers`)
- Remove `albumentations` from requirements (not used in production code)
- Product code stays fully proprietary at no ongoing cost

**Option B — Purchase commercial licenses**
- Ultralytics Enterprise License: ~$1,000–5,000/year
- Albumentations commercial license: separate pricing
- Keep AGPL dependencies as-is, proprietary code stays closed

**Option C — Release the product as open source**
- Publish under AGPL-3.0 or open-core model
- Revenue from services, support, and enterprise features
- Maximises community visibility

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

### Recommended deployment options

**Option 1 — On-premise Docker** *(primary, already prototyped)*
- Extend the existing `docker-compose.yml`
- Client installs on a local server or company VM
- Multi-user, browser access from internal network
- Easy to update; no data leaves the network
- Requires minimal IT skills to install

**Option 2 — Standalone Desktop** *(for individual practitioners)*
- Windows/macOS installer (.exe / .dmg)
- FastAPI backend bundled with PyInstaller + Electron UI
- Fully local, works offline, no network dependencies
- Suitable for single-examiner licensing

**Option 3 — SaaS Cloud** *(optional, future)*
- Hosted on AWS / Azure / GCP with free tier + paid plans
- No installation, automatic updates, recurring revenue
- More complex GDPR compliance; expect resistance from forensic clients
- Add as a secondary channel once on-premise is established

---

## Development Roadmap

### Phase 0 — Prerequisites (est. 1–2 months)

- [ ] Replace `ultralytics` with AGPL-free detector (RT-DETR via `transformers`)
- [ ] Remove `albumentations` from requirements
- [ ] Choose web stack: **FastAPI** (backend) + **React or Vue** (frontend), or Django full-stack
- [ ] Choose database: **PostgreSQL** for cases and metadata, **MinIO** (S3-compatible) for image files
- [ ] Define commercial licensing model

### Phase 1 — MVP (est. 3–4 months)

Professional core that replaces the Jupyter notebooks.

- **Case management:** create case, upload documents, attach reference samples
- **4 AI engines** integrated in the UI: HTR, signature verification, signature detection, graphological analysis
- **User authentication** with roles (admin, examiner, viewer)
- **Immutable audit log** of every operation (forensic requirement)
- **PDF report** generated automatically with images, metrics, and legal disclaimer
- **Deployment:** Docker Compose on-premise, installable in under 10 minutes

### Phase 2 — Mature Product (est. 3–4 months)

- Batch processing: automatic analysis of document archives
- Multi-sample writer identification with comparative interface
- Side-by-side sample comparison with annotations
- Data export (CSV, JSON) for integration with third-party systems
- AI model updates without reinstallation
- End-user documentation and examiner manual

### Phase 3 — Enterprise (est. 4–6 months)

- SSO / LDAP / Active Directory integration
- Multi-tenancy (multiple organisations on the same instance)
- Public REST API (for integration with court or banking systems)
- Fine-tuning on client-proprietary datasets
- Usage analytics dashboard
- SLA support agreements and enterprise contracts

---

## Recommended Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| AI / ML | PyTorch, Transformers, OpenCV | Unchanged from labs |
| Backend API | FastAPI | Async, automatic OpenAPI docs |
| Frontend | React + Tailwind CSS | or Vue 3 |
| Database | PostgreSQL | Cases, users, audit log |
| File storage | MinIO (S3-compatible) | Documents and images, on-premise |
| Auth | JWT + bcrypt | or Keycloak for enterprise SSO |
| PDF reports | WeasyPrint or ReportLab | Forensic report generation |
| Container | Docker Compose | Extend existing setup |
| CI/CD | GitHub Actions | Build, test, automated releases |

---

## Licensing Model for the Product

### Recommended: Open-Core

| Edition | License | Target |
|---|---|---|
| **Community** | Apache 2.0, open source | Individual researchers, academia |
| **Professional** | Commercial, closed source | Examiners, law firms |
| **Enterprise** | Commercial + SLA contract | Courts, banks, government |

**Rationale:** A Community Edition on GitHub maximises visibility and builds reputation in the forensic and AI communities. Professional and Enterprise tiers generate revenue through case management, reporting, audit logging, and support — features that matter to paying clients but are overkill for the open-source demo use case.

### Customer pricing (indicative)

| Plan | Target | Model |
|---|---|---|
| **Solo** | Single examiner | Annual licence, 1 installation |
| **Studio** | 2–10 users | Annual licence per user |
| **Enterprise** | Courts, banks | Contract + SLA + fine-tuning |
| **SaaS** *(future)* | Small firms | Monthly subscription / pay-per-use |

---

## Open Questions for Future Decisions

- Confirm that RT-DETR (or alternative) covers the Lab 04 signature detection use case adequately
- Verify Ultralytics Enterprise License pricing if Option B is preferred over Option A
- Decide frontend framework (React vs Vue) before starting Phase 1
- Assess whether a UX/UI designer is needed for the case management interface
- Evaluate whether to seek grant funding (EU Horizon, PNRR) for forensic AI tooling
