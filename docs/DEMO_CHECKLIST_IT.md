# GraphoLab — Lista della Spesa per la Demo

Tutto ciò che serve per eseguire tutti e otto i notebook GraphoLab dall'inizio alla fine: quali modelli AI vengono scaricati automaticamente e quali immagini campione devi fornire tu.

---

## Riepilogo Rapido

| Cosa | Dove | Note |
|------|------|------|
| Ambiente Python | locale o Docker | vedi [NOTEBOOKS_GUIDE_IT.md](NOTEBOOKS_GUIDE_IT.md) |
| `requirements.txt` installato | — | `pip install -r requirements.txt` |
| Pesi SigNet | `models/signet.pth` | download manuale — vedi sezione Lab 03 |
| Immagini campione | `data/samples/` | vedi le sezioni per lab qui sotto |
| Modelli AI | scaricati automaticamente | connessione internet necessaria al primo avvio |

---

## Modelli AI — Scaricati Automaticamente

Tutti i modelli Hugging Face vengono recuperati al primo avvio e memorizzati nella cache locale (o nel volume Docker `grapholab-hf-cache`).

| Modello | Scaricato da | Dimensione | Cache |
|---------|-------------|-----------|-------|
| **TrOCR** (`microsoft/trocr-base-handwritten`) | `transformers` | ~400 MB | `~/.cache/huggingface/` |
| **EasyOCR** (modelli italiano + inglese) | `easyocr` | ~100 MB | `~/.EasyOCR/` |
| **Rilevatore firme YOLOv8s** (`tech4humans/yolov8s-signature-detector`) | `huggingface_hub` | ~22 MB | `~/.cache/huggingface/` |
| **WikiNEural NER** (`Babelscape/wikineural-multilingual-ner`) | `transformers` | ~700 MB | `~/.cache/huggingface/` |
| **dots.ocr** (`rednote-hilab/dots.ocr`) | `transformers` | ~3,5 GB (bf16) / ~7 GB (fp32 CPU) | `~/.cache/huggingface/` |

> **È necessaria una connessione internet al primo avvio dei Lab 02, 04, 07 e 08.** Le esecuzioni successive utilizzano i modelli in cache.
>
> **dots.ocr (Lab 08) richiede anche un `git clone` una tantum** — vedi la cella di installazione nel notebook.

## Modelli AI — Download Manuale Richiesto

| Modello | File | Dimensione | Fonte |
|---------|------|-----------|-------|
| **SigNet** (pre-addestrato su GPDS) | `models/signet.pth` | ~63 MB | [luizgh/sigver](https://github.com/luizgh/sigver) |

Scarica `signet.pth` dal repository sigver e inseriscilo nella directory `models/` prima di eseguire il Lab 03.

---

## Immagini Campione — Cosa Devi Fornire

Inserisci tutte le immagini in `data/samples/`. Quando le immagini reali mancano, vengono generate automaticamente immagini sintetiche di fallback, quindi i notebook vengono sempre eseguiti — ma i risultati su dati sintetici non hanno valore forense reale.

### Lab 01 — Introduzione
**Nessuna immagine richiesta.** Notebook solo testuale.

---

### Lab 02 — Riconoscimento Testo Manoscritto (TrOCR)

| File | Descrizione |
|------|-------------|
| `handwritten_text_01.png` | Una riga di testo manoscritto |
| `handwritten_text_02.png` | (opzionale) Secondo campione su riga singola |
| `handwritten_multiline_01.png` | Un documento manoscritto multiriga (per la pipeline HTR→NER) |

**Requisiti:**
- Scansione o foto nitida di testo manoscritto
- Risoluzione consigliata: 300 DPI o superiore
- Sfondo bianco o chiaro, inchiostro scuro
- TrOCR è un modello a livello di riga; le immagini multiriga vengono suddivise automaticamente per proiezione orizzontale prima dell'inferenza

**Confronto con trascrizione nota (opzionale):** se disponi della trascrizione esatta del testo manoscritto, puoi calcolare il Character Error Rate (CER) nella sezione opzionale del Lab 02.

---

### Lab 03 — Verifica della Firma (SigNet)

| File | Descrizione |
|------|-------------|
| `genuine_N_1.png` | **Firma di riferimento** — autentica nota (scrittore N, campione 1) |
| `genuine_N_2.png` | Seconda firma autentica dello stesso scrittore |
| `forged_N_M.png` | Firma contraffatta (scrittore N, falsificazione M) |

Ripeti per ogni scrittore che vuoi dimostrare (es. N = 1, 2, 3, …).

**Requisiti:**
- Firme isolate (nessun testo del documento circostante)
- Sfondo bianco o chiaro, inchiostro scuro
- Qualità di scansione uniforme tra i campioni della stessa persona
- Risoluzione consigliata: 300 DPI o superiore

> **Campioni demo pre-selezionati:** il repository include coppie curate dal database di firme **CEDAR**. Queste coppie sono state pre-scansionate con SigNet per verificare che il modello classifichi correttamente la contraffazione (distanza coseno > 0.35). Gli scrittori 1–5 corrispondono agli scrittori CEDAR 51, 26, 34, 32 e 21.

> **Pesi SigNet richiesti:** scarica `models/signet.pth` da [luizgh/sigver](https://github.com/luizgh/sigver) prima di eseguire questo lab.

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

### Lab 07 — Riconoscimento Entità Nominate (NER)

**Nessun file immagine richiesto.** Il modello NER opera direttamente su stringhe di testo.

- **Demo 1 & 2:** testi di esempio italiani e inglesi inclusi nel notebook — nessun file necessario.
- **Demo 3 (pipeline HTR→NER):** carica `handwritten_multiline_01.png` (condiviso con il Lab 02).

Il modello `Babelscape/wikineural-multilingual-ner` (~700 MB) viene scaricato automaticamente al primo avvio. Supporta 9 lingue tra cui italiano e inglese.

---

### Lab 08 — dots.ocr (OCR con Vision-Language Model)

| File | Descrizione |
| ---- | ----------- |
| `writer_00/sample_000.png` | Campione singolo writer_00 (condiviso con Lab 05) |
| `testamento_writer00.png` | Documento testamento completo — generare con `scripts/create_testamento_writer00.py` |
| `lorella/*.png` | (opzionale) Campioni di scrittura reale |

**Requisiti:**

- Primo avvio: connessione internet per il download del modello (~3,5 GB bf16 o ~7 GB fp32 su CPU)
- Su CPU: ~7 GB di RAM libera; 2–5 min per immagine
- Su GPU: ≥4 GB VRAM raccomandati

**Installazione una tantum (prima del primo avvio):**

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git DotsOCR
pip install -e DotsOCR
pip install qwen_vl_utils accelerate
```

---

## Riepilogo delle Convenzioni di Denominazione

```
data/samples/
  handwritten_text_01.png          # Lab 02, 06
  handwritten_text_02.png          # Lab 02, 06 (opzionale)
  handwritten_multiline_01.png     # Lab 02, 07 (HTR multiriga + pipeline NER)
  genuine_1_1.png                  # Lab 03 — scrittore 1, riferimento
  genuine_1_2.png                  # Lab 03 — scrittore 1, secondo campione autentico
  forged_1_1.png                   # Lab 03 — scrittore 1, contraffatta
  genuine_2_1.png                  # Lab 03 — scrittore 2, riferimento
  ...
  document_with_signature_01.png   # Lab 04
  writer_01/sample_01.png          # Lab 05
  writer_01/sample_02.png          # Lab 05
  ...
```

---

## Demo Minima Funzionante (5 immagini)

Per una demo rapida che copra i Lab 02, 03, 04, 06 e 07 con un insieme minimo di immagini:

1. `handwritten_text_01.png` — per i Lab 02 e 06
2. `handwritten_multiline_01.png` — per la pipeline HTR→NER del Lab 07
3. `genuine_1_1.png` — firma di riferimento
4. `forged_1_1.png` — firma contraffatta
5. `document_with_signature_01.png` — pagina documento per il Lab 04

Il Lab 01 non richiede nulla. Il Lab 05 richiede le sottodirectory per scrittore (non coperte da questo set minimo). Le Demo 1 & 2 del Lab 07 non richiedono alcun file.

---

## Checklist Prima di Avviare i Laboratori

- [ ] Ambiente Python creato e `requirements.txt` installato
- [ ] Connessione internet disponibile (download modelli al primo avvio: TrOCR ~400 MB, EasyOCR ~100 MB, WikiNEural NER ~700 MB, YOLOv8 ~22 MB, dots.ocr ~3,5 GB)
- [ ] `models/signet.pth` scaricato da [luizgh/sigver](https://github.com/luizgh/sigver)
- [ ] Directory `data/samples/` presente
- [ ] Immagini di testo manoscritto inserite (`handwritten_text_*.png`, `handwritten_multiline_01.png`)
- [ ] Immagini delle firme inserite (`genuine_N_M.png`, `forged_N_M.png`)
- [ ] Scansione del documento inserita (`document_with_signature_*.png`)
- [ ] Sottodirectory degli scrittori popolate (`writer_XX/sample_YY.png`) — per il Lab 05
- [ ] JupyterLab avviato (`jupyter lab` oppure `docker compose up jupyter`)
