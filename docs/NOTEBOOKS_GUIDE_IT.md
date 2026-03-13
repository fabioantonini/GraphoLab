# GraphoLab — Guida ai Laboratori

Guida pratica ai laboratori dimostrativi di GraphoLab sulla grafologia forense assistita dall'AI.

---

## Panoramica

La grafologia forense è l'esame scientifico della scrittura a mano e delle firme a supporto delle indagini legali. L'AI e il machine learning possono automatizzare e scalare molti compiti che gli esperti svolgono tradizionalmente in modo manuale:

| Compito | Approccio AI | Applicazione Forense |
|---------|-------------|----------------------|
| Trascrizione di testo manoscritto | Transformer OCR (TrOCR) | Lettere anonime, documenti storici |
| Autenticità della firma | Siamese Neural Network | Assegni, contratti, testamenti |
| Localizzazione firma nei documenti | Object Detection (YOLOv8) | Pipeline di analisi documentale |
| Identificazione dello scrittore | Estrazione feature + classificatore | Paternità contestata |
| Analisi caratteristiche grafologiche | OpenCV + ML | Profiling, analisi comparativa |

---

## Lab 01 — Introduzione: AI e Grafologia Forense

**File:** `notebooks/01_intro_forensic_graphology.ipynb`

Notebook in stile presentazione, senza codice eseguibile. Tratta:

- Cos'è la grafologia forense e perché è importante
- I limiti dell'analisi puramente manuale
- Come l'AI/ML può potenziare il lavoro dell'esperto
- Mappa concettuale dell'intera pipeline: acquisizione → preprocessing → analisi AI → referto forensico
- Panoramica di tutti i laboratori successivi

**Prerequisiti:** Nessuno.
**Destinatari:** Tutti i livelli, inclusi stakeholder non tecnici.

---

## Lab 02 — Riconoscimento del Testo Manoscritto (HTR/OCR)

**File:** `notebooks/02_handwritten_ocr_trocr.ipynb`

Utilizza **TrOCR** (`microsoft/trocr-base-handwritten` su Hugging Face) per trascrivere automaticamente testo manoscritto da immagini.

**Cosa imparerai:**
- Come funziona l'OCR basato su Transformer (encoder visivo BEiT + decoder testuale RoBERTa)
- Come caricare ed eseguire un modello HTR pre-addestrato tramite la libreria `transformers`
- Come preprocessare immagini di testo manoscritto per il modello
- Come interpretare e visualizzare l'output della trascrizione

**Flusso della demo:**
1. Caricare un'immagine di testo manoscritto da `data/samples/`
2. Eseguire l'inferenza con TrOCR
3. Visualizzare l'immagine originale affiancata al testo trascritto
4. (Opzionale) Confrontare con la trascrizione di riferimento e calcolare il CER (Character Error Rate)

**Casi d'uso forensi:**
- Trascrizione automatica di lettere minatorie anonime
- Digitalizzazione di documenti giudiziari manoscritti storici
- Fase di pre-elaborazione per pipeline di identificazione dell'autore

**Prerequisiti:** `transformers`, `torch`, `Pillow`

---

## Lab 03 — Verifica dell'Autenticità della Firma

**File:** `notebooks/03_signature_verification_siamese.ipynb`

Utilizza una **Siamese Neural Network** (architettura SigNet) per confrontare due immagini di firme e determinare se la firma in esame è autentica o contraffatta.

**Cosa imparerai:**
- Il paradigma della rete siamese per il one-shot similarity learning
- Come SigNet codifica le immagini di firme in vettori di feature
- Come calcolare uno score di similarità (o distanza) tra due firme
- Come impostare una soglia decisionale per la classificazione autentica / contraffatta

**Flusso della demo:**
1. Caricare una firma di riferimento e una firma in esame da `data/samples/`
2. Estrarre gli embedding di feature con il codificatore SigNet
3. Calcolare lo score di similarità coseno
4. Visualizzare: verdetto autentica / contraffatta + score di confidenza

**Casi d'uso forensi:**
- Verifica di firme su assegni bancari
- Autenticazione di firme su contratti, testamenti e atti legali
- Rilevamento di firme tracciate o riprodotte digitalmente

**Prerequisiti:** `torch`, `torchvision`, `Pillow`

**Riferimento:** [luizgh/sigver](https://github.com/luizgh/sigver) — implementazione SigNet e pesi pre-addestrati.

---

## Lab 04 — Rilevamento Firma nei Documenti (YOLOv8)

**File:** `notebooks/04_signature_detection_yolo.ipynb`

Utilizza un modello **YOLOv8** fine-tuned per il rilevamento di firme (`tech4humans/yolov8s-signature-detector` su Hugging Face) per localizzare automaticamente le firme all'interno di documenti scansionati.

**Cosa imparerai:**
- Come funziona la rilevazione di oggetti YOLO nel contesto dell'analisi documentale
- Come caricare un modello Hugging Face con la libreria `ultralytics`
- Come eseguire l'inferenza su immagini di documenti e interpretare i bounding box
- Come visualizzare le regioni rilevate e ritagliarle per l'elaborazione successiva

**Flusso della demo:**
1. Caricare un'immagine di documento scansionato da `data/samples/`
2. Eseguire l'inferenza YOLOv8
3. Disegnare i bounding box attorno alle firme rilevate
4. Ritagliare e salvare ciascuna firma rilevata per l'uso nel Lab 03

**Casi d'uso forensi:**
- Estrazione automatica di firme da documenti legali multi-pagina
- Primo passo di una pipeline: rileva → estrai → verifica
- Screening di grandi archivi documentali per la presenza di firme

**Prerequisiti:** `ultralytics`, `opencv-python`, `Pillow`

---

## Lab 05 — Identificazione dello Scrittore

**File:** `notebooks/05_writer_identification.ipynb`

Confronta le caratteristiche stilistiche di un campione manoscritto anonimo con un insieme di campioni di riferimento noti per attribuire la paternità del documento.

**Cosa imparerai:**
- Come estrarre feature stilistiche della scrittura (descrittori di texture, statistiche dei tratti)
- Come costruire un semplice classificatore per l'identificazione dello scrittore con scikit-learn
- Come valutare l'accuratezza dell'identificazione tramite cross-validation
- Come presentare i risultati come lista ordinata di autori candidati con score di confidenza

**Flusso della demo:**
1. Caricare un piccolo set di campioni di riferimento (autori noti) da `data/samples/`
2. Estrarre le feature da ciascun campione
3. Addestrare o caricare un classificatore pre-addestrato
4. Caricare un campione anonimo → lista ordinata di candidati con score di similarità

**Casi d'uso forensi:**
- Attribuzione di lettere minatorie anonime
- Verifica della paternità di documenti contestati
- Ricerca sulla provenienza di documenti storici

**Prerequisiti:** `torch`, `torchvision`, `scikit-learn`, `Pillow`, `numpy`

**Nota sul dataset:** Il [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) è il benchmark standard. Un piccolo sottoinsieme pubblico è incluso in `data/samples/` a scopo dimostrativo.

---

## Lab 06 — Analisi delle Caratteristiche Grafologiche

**File:** `notebooks/06_graphological_feature_analysis.ipynb`

Estrae e visualizza automaticamente le caratteristiche grafologiche di un campione di testo manoscritto utilizzando la computer vision classica e l'elaborazione del segnale.

**Cosa imparerai:**
- Come segmentare la scrittura in parole e caratteri con OpenCV
- Come misurare: angolo di inclinazione delle lettere, spaziatura parole/caratteri, altezza e larghezza delle lettere, pressione del tratto (distribuzione dell'intensità dei pixel)
- Come costruire una dashboard visiva delle metriche grafologiche
- Come confrontare due campioni ed evidenziare le differenze

**Flusso della demo:**
1. Caricare un'immagine di testo manoscritto da `data/samples/`
2. Pre-elaborare (binarizzare, denoisare, raddrizzare)
3. Segmentare in righe, parole e caratteri
4. Calcolare e visualizzare le metriche con annotazioni sull'immagine
5. (Opzionale) Confrontare due campioni affiancati

**Casi d'uso forensi:**
- Supporto alla testimonianza peritale con misurazioni oggettive e riproducibili
- Rilevamento di indicatori di stress nella scrittura (variazione della pressione del tratto)
- Analisi comparativa tra un campione di riferimento e un documento contestato

**Prerequisiti:** `opencv-python`, `numpy`, `matplotlib`, `scipy`

---

## Demo Interattiva (Gradio)

**File:** `app/grapholab_demo.py`

Un'applicazione Gradio multi-tab accessibile da browser che aggrega le quattro principali funzionalità AI:

| Tab | Funzionalità |
|-----|-------------|
| Handwritten OCR | Carica un'immagine → testo trascritto |
| Signature Verification | Carica due firme → verdetto autentica / contraffatta |
| Signature Detection | Carica un documento → immagine annotata con firme rilevate |
| Graphological Analysis | Carica testo manoscritto → dashboard di metriche visive |

**Avvio in locale:**

```bash
# Crea e attiva un virtual environment (consigliato)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app/grapholab_demo.py
# Apri http://localhost:7860
```

---

## Esecuzione con Docker

```bash
# JupyterLab su http://localhost:8888  (token: grapholab)
docker compose up jupyter

# Demo Gradio su http://localhost:7860
docker compose up gradio

# Entrambi i servizi insieme
docker compose up
```

La cache dei modelli Hugging Face è memorizzata in un volume Docker dedicato (`grapholab-hf-cache`) e condivisa tra i due servizi. I modelli vengono scaricati una sola volta.

---

## Dati di Esempio

Le immagini di esempio si trovano in `data/samples/`:

| File | Usato in |
|------|---------|
| `handwritten_text_*.png` | Lab 02, 05, 06 |
| `signature_genuine_*.png` | Lab 03, 04 |
| `signature_forged_*.png` | Lab 03 |
| `document_with_signature_*.png` | Lab 04 |

È possibile sostituire o integrare questi file con immagini proprie per sperimentare con casi reali.
