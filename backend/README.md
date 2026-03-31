# GraphoLab — Backend (FastAPI)

REST API che espone tutti i moduli AI di `core/` attraverso 35 endpoint HTTP.
Pensato per essere consumato dal frontend React (in sviluppo in `feature/frontend`).

---

## Testare in locale — senza Docker

### Prerequisiti

```bash
# Installa le dipendenze backend (una volta sola)
venv\Scripts\pip install -r requirements-backend.txt
```

### Opzione A — SQLite (più rapida, zero configurazione)

Ideale per sviluppo e test degli endpoint API senza installare nulla di extra.

1. Copia il file di esempio e imposta SQLite:

```bash
copy backend\.env.example backend\.env
```

2. Modifica `backend\.env` e sostituisci la riga `DATABASE_URL`:

```
DATABASE_URL=sqlite+aiosqlite:///./grapholab_dev.db
```

> Aggiungi anche `aiosqlite` al venv se non presente:
> `venv\Scripts\pip install aiosqlite`

3. Avvia il backend:

```bash
venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

4. Apri la **Swagger UI** per testare tutti gli endpoint dal browser:

```
http://localhost:8000/docs
```

Con questa configurazione funzionano: autenticazione, gestione utenti, CRUD progetti,
analisi AI. **Non** funziona l'upload file su MinIO (richiede il servizio MinIO).

---

### Opzione B — PostgreSQL + MinIO locali (ambiente completo)

Se hai PostgreSQL e MinIO già installati in locale (o vuoi avviarli con Docker),
configura `backend\.env` con i valori reali e avvia normalmente:

```bash
venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

---

## Testare in locale — con Docker Compose

Avvia solo i servizi necessari (senza jupyter/gradio):

```bash
# Solo infrastruttura (postgres + minio)
docker compose up postgres minio

# Infrastruttura + backend API
docker compose up postgres minio backend

# Stack completo (aggiunge anche ollama + gradio + jupyter)
docker compose up
```

| Servizio | URL locale |
|---|---|
| **Backend API** | http://localhost:8000/docs |
| **MinIO console** | http://localhost:9001 (user: grapholab / grapholab123) |
| **PostgreSQL** | localhost:5432 (db: grapholab) |
| **Ollama** | http://localhost:11434 |
| Gradio demo | http://localhost:7860 |
| JupyterLab | http://localhost:8888 (token: grapholab) |

---

## Workflow di test consigliato

1. Avvia il backend (opzione A o B)
2. Vai su `http://localhost:8000/docs`
3. **Crea un utente admin** — `POST /users/` (richiede token, usa prima il login di un admin già esistente nel DB, oppure crea manualmente il primo utente con uno script)
4. **Login** — `POST /auth/login` → ottieni `access_token`
5. Clicca **Authorize** in Swagger UI e incolla il token
6. Testa gli endpoint in ordine: crea progetto → carica documento → lancia analisi

### Script per creare il primo utente admin

```python
# Esegui una volta per inizializzare il DB con un admin
# python -m backend.scripts.create_admin  (da aggiungere)

# Oppure direttamente:
import asyncio, sys
sys.path.insert(0, '.')
from backend.database import AsyncSessionLocal, init_db
from backend.models.user import User, Role
from backend.auth.password import hash_password

async def main():
    await init_db()
    async with AsyncSessionLocal() as db:
        admin = User(
            email="admin@grapholab.local",
            full_name="Administrator",
            hashed_password=hash_password("changeme"),
            role=Role.admin,
        )
        db.add(admin)
        await db.commit()
        print("Admin creato: admin@grapholab.local / changeme")

asyncio.run(main())
```

---

## Struttura

```
backend/
├── main.py                  # FastAPI app, CORS, lifespan, health check
├── config.py                # Settings (env vars / .env file)
├── database.py              # SQLAlchemy async engine + get_db()
├── auth/
│   ├── jwt.py               # Creazione e verifica token JWT
│   ├── password.py          # bcrypt hash/verify
│   └── dependencies.py      # get_current_user(), require_role()
├── models/
│   ├── user.py              # User, Organization, Role
│   └── project.py           # Project, Document, Analysis
├── routers/
│   ├── auth.py              # POST /auth/login|refresh|logout
│   ├── users.py             # CRUD /users/ + /users/me
│   ├── projects.py          # CRUD /projects/ + upload documenti
│   ├── analysis.py          # POST /analysis/{htr,sig,ner,writer,grapho,pipeline,dating}
│   └── rag.py               # /rag/chat (SSE) + /rag/docs
└── storage/
    └── minio_client.py      # Async wrapper MinIO SDK
```

Tutto il processing AI è in `core/` — questo layer è solo HTTP + persistenza.

---

## Variabili d'ambiente

Copia `backend/.env.example` in `backend/.env` e personalizza:

| Variabile | Default | Note |
|---|---|---|
| `SECRET_KEY` | `change_me_...` | **Cambia in produzione** (`openssl rand -hex 32`) |
| `DATABASE_URL` | PostgreSQL asyncpg | Sostituisci con `sqlite+aiosqlite:///./dev.db` per test rapidi |
| `MINIO_ENDPOINT` | `localhost:9000` | |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `DEBUG` | `false` | `true` per query SQL nei log |
