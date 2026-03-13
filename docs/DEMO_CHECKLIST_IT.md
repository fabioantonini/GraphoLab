# GraphoLab — Lista della Spesa per la Demo

Tutto ciò che serve per eseguire tutti e sei i notebook GraphoLab dall'inizio alla fine: quali modelli AI vengono scaricati automaticamente e quali immagini campione devi fornire tu.

---

## Riepilogo Rapido

| Cosa | Dove | Note |
|------|------|------|
| Ambiente Python | locale o Docker | vedi [NOTEBOOKS_GUIDE_IT.md](NOTEBOOKS_GUIDE_IT.md) |
| `requirements.txt` installato | — | `pip install -r requirements.txt` |
| Immagini campione | `data/samples/` | vedi le sezioni per lab qui sotto |
| Modelli AI | scaricati automaticamente | connessione internet necessaria al primo avvio |

---

## Modelli AI — Scaricati Automaticamente

Non è necessario scaricare pesi manualmente. Tutti i modelli vengono recuperati al primo avvio e memorizzati nella cache locale (o nel volume Docker `grapholab-hf-cache`).

| Modello | Scaricato da | Dimensione | Cache |
|---------|-------------|-----------|-------|
| **ResNet-18** (pesi ImageNet) | `torchvision` | ~45 MB | `~/.cache/torch/hub/` |
| **TrOCR** (`microsoft/trocr-base-handwritten`) | `transformers` | ~400 MB | `~/.cache/huggingface/` |
| **Rilevatore firme YOLOv8s** (`tech4humans/yolov8s-signature-detector`) | `huggingface_hub` | ~22 MB | `~/.cache/huggingface/` |

> **È necessaria una connessione internet al primo avvio dei Lab 02, 03 e 04.** Le esecuzioni successive utilizzano i modelli in cache.

---

## Immagini Campione — Cosa Devi Fornire

Inserisci tutte le immagini in `data/samples/`. Quando le immagini reali mancano, vengono generate automaticamente immagini sintetiche di fallback, quindi i notebook vengono sempre eseguiti — ma i risultati su dati sintetici non hanno valore forense reale.

### Lab 01 — Introduzione
**Nessuna immagine richiesta.** Notebook solo testuale.

---

### Lab 02 — Riconoscimento Testo Manoscritto (TrOCR)

| File | Descrizione |
|------|-------------|
| `handwritten_text_01.png` | Una riga o paragrafo di testo manoscritto |
| `handwritten_text_02.png` | (opzionale) Secondo campione per la demo in batch |

**Requisiti:**
- Scansione o foto nitida di testo manoscritto
- Risoluzione consigliata: 300 DPI o superiore
- Sfondo bianco o chiaro, inchiostro scuro
- Una singola riga o un breve paragrafo funzionano meglio

**Confronto con trascrizione nota (opzionale):** se disponi della trascrizione esatta del testo manoscritto, puoi calcolare il Character Error Rate (CER) nella sezione opzionale del Lab 02.

---

### Lab 03 — Verifica della Firma (ResNet-18)

| File | Descrizione |
|------|-------------|
| `signature_genuine_01.png` | **Firma di riferimento** — autentica nota |
| `signature_genuine_02.png` | Seconda firma autentica della stessa persona |
| `signature_forged_01.png` | Firma contraffatta o contestata |

Per il **grafico di distribuzione** (Demo 5), aggiungi altri file con lo stesso schema di denominazione:
- `signature_genuine_03.png`, `signature_genuine_04.png`, … (altri campioni autentici)
- `signature_forged_02.png`, `signature_forged_03.png`, … (altri campioni contraffatti)

**Requisiti:**
- Firme isolate (nessun testo del documento circostante)
- Sfondo bianco o chiaro, inchiostro scuro
- Qualità di scansione uniforme tra i campioni della stessa persona
- Risoluzione consigliata: 300 DPI o superiore

> **Minimo per una demo significativa:** 1 firma di riferimento + 1 copia autentica + 1 contraffatta. Il modello (pesi ImageNet di ResNet-18) funziona immediatamente — nessun addestramento richiesto.

---

### Lab 04 — Rilevamento Firma nei Documenti (YOLOv8)

| File | Descrizione |
|------|-------------|
| `document_with_signature_01.png` | Una pagina di documento scansionata con almeno una firma |

**File aggiuntivi opzionali:** `document_with_signature_02.png`, `document_with_signature_03.png`, …

**Requisiti:**
- Immagine dell'intera pagina del documento (non una firma già ritagliata)
- Il modello gestisce pagine con più firme
- Risoluzione consigliata: 200–300 DPI
- Funziona su contratti, lettere, moduli, assegni bancari

> **Output:** le firme rilevate vengono ritagliate e salvate come `detected_signature_N.png` in `data/samples/`. Questi ritagli possono essere usati direttamente come input per il Lab 03.

---

### Lab 05 — Identificazione dello Scrittore

Organizzato in sottodirectory per scrittore all'interno di `data/samples/`:

```
data/samples/
  writer_01/
    sample_01.png
    sample_02.png
    sample_03.png
    sample_04.png
    sample_05.png
  writer_02/
    sample_01.png
    ...
  writer_03/
    sample_01.png
    ...
```

**Requisiti:**
- Minimo **3 scrittori** (di più = maggiore accuratezza)
- Minimo **5 campioni per scrittore** (il notebook usa la cross-validation leave-one-out)
- Ogni campione: alcune righe di testo manoscritto continuo
- Condizioni di scansione uniformi per tutti i campioni
- Risoluzione consigliata: 300 DPI

> **Nota sull'addestramento:** il Lab 05 addestra un classificatore SVM leggero sui campioni forniti ad ogni esecuzione del notebook. Non viene utilizzato nessun modello pre-addestrato per l'identificazione dello scrittore — i tuoi campioni sono i dati di addestramento.

---

### Lab 06 — Analisi delle Caratteristiche Grafologiche

Riutilizza le immagini di testo manoscritto del Lab 02:

| File | Descrizione |
|------|-------------|
| `handwritten_text_01.png` | Campione principale per l'estrazione delle caratteristiche |
| `handwritten_text_02.png` | (opzionale) Secondo campione per il confronto affiancato |

Non sono necessari file aggiuntivi se i campioni del Lab 02 sono già presenti.

---

## Riepilogo delle Convenzioni di Denominazione

```
data/samples/
  handwritten_text_01.png          # Lab 02, 06
  handwritten_text_02.png          # Lab 02, 06 (opzionale)
  signature_genuine_01.png         # Lab 03 — riferimento
  signature_genuine_02.png         # Lab 03 — copia autentica
  signature_genuine_03.png         # Lab 03 — grafico distribuzione (opzionale)
  signature_forged_01.png          # Lab 03 — contraffatta / contestata
  signature_forged_02.png          # Lab 03 — grafico distribuzione (opzionale)
  document_with_signature_01.png   # Lab 04
  writer_01/sample_01.png          # Lab 05
  writer_01/sample_02.png          # Lab 05
  ...
```

---

## Demo Minima Funzionante (5 immagini)

Per una demo rapida che copra i Lab 02, 03, 04 e 06 con un insieme minimo di immagini:

1. `handwritten_text_01.png` — per i Lab 02 e 06
2. `signature_genuine_01.png` — firma di riferimento
3. `signature_genuine_02.png` — seconda firma autentica
4. `signature_forged_01.png` — firma contraffatta
5. `document_with_signature_01.png` — pagina documento per il Lab 04

Il Lab 01 non richiede nulla. Il Lab 05 richiede le sottodirectory per scrittore (non coperte da questo set minimo).

---

## Checklist Prima di Avviare i Laboratori

- [ ] Ambiente Python creato e `requirements.txt` installato
- [ ] Connessione internet disponibile (download modelli al primo avvio)
- [ ] Directory `data/samples/` presente
- [ ] Immagini di testo manoscritto inserite (`handwritten_text_*.png`)
- [ ] Immagini delle firme inserite (`signature_genuine_*.png`, `signature_forged_*.png`)
- [ ] Scansione del documento inserita (`document_with_signature_*.png`)
- [ ] Sottodirectory degli scrittori popolate (`writer_XX/sample_YY.png`) — per il Lab 05
- [ ] JupyterLab avviato (`jupyter lab` oppure `docker compose up jupyter`)
