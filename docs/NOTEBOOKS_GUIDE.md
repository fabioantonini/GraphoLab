# GraphoLab — Notebooks Guide

A practical guide to the GraphoLab demo labs on AI-assisted forensic graphology (8 labs).

---

## Overview

Forensic graphology is the scientific examination of handwriting and signatures to support legal investigations. AI and machine learning can automate and scale several tasks that experts traditionally perform manually:

| Task | AI Approach | Forensic Application |
|------|-------------|----------------------|
| Handwritten text transcription | Transformer OCR (TrOCR / EasyOCR) | Anonymous letters, historical documents |
| Signature authenticity | Siamese Neural Network (SigNet) | Checks, contracts, wills |
| Signature location in documents | Object Detection (YOLOv8) | Document processing pipelines |
| Writer identification | Feature extraction + classifier | Disputed authorship |
| Graphological feature analysis | OpenCV + ML | Profiling, comparative analysis |
| Named entity recognition | Token classification (BERT-NER) | People, places, orgs in documents |
| Deep OCR (Italian cursive) | Vision-Language Model (dots.ocr 1.7B) | Wills, cursive handwriting, complex layouts |

---

## Lab 01 — Introduction: AI and Forensic Graphology

**File:** `notebooks/01_intro_forensic_graphology.ipynb`

A presentation-style notebook with no executable code. Covers:

- What is forensic graphology and why it matters
- The limits of purely manual analysis
- How AI/ML can augment expert analysis
- Concept map of the full pipeline: acquisition → preprocessing → AI analysis → forensic report
- Overview of all subsequent labs

**Prerequisites:** None.
**Audience:** All levels, including non-technical stakeholders.

---

## Lab 02 — Handwritten Text Recognition (HTR/OCR)

**File:** `notebooks/02_handwritten_ocr_trocr.ipynb`

Uses **TrOCR** (`microsoft/trocr-base-handwritten` on Hugging Face) to automatically transcribe handwritten text from images.

**What you will learn:**
- How Transformer-based OCR works (BEiT vision encoder + RoBERTa text decoder)
- How to load and run a pre-trained HTR model via the `transformers` library
- How to preprocess handwritten images for the model
- How to interpret and visualise the transcription output

**Demo flow:**
1. Load a sample handwritten image from `data/samples/`
2. Run TrOCR inference
3. Display the original image alongside the transcribed text
4. (Optional) Compare against a ground-truth transcript and compute CER (Character Error Rate)

**Forensic use cases:**
- Automatic transcription of anonymous threatening letters
- Digitisation of historical handwritten court documents
- Pre-processing step for author identification pipelines

**Prerequisites:** `transformers`, `torch`, `Pillow`

---

## Lab 03 — Signature Authenticity Verification

**File:** `notebooks/03_signature_verification_siamese.ipynb`

Uses a **Siamese Neural Network** (SigNet architecture) to compare two signature images and determine whether the questioned signature is genuine or forged.

**What you will learn:**
- The Siamese network paradigm for one-shot similarity learning
- How SigNet encodes signature images into feature vectors
- How to compute a similarity (or distance) score between two signatures
- Setting a decision threshold for genuine / forged classification

**Demo flow:**
1. Load a reference signature and a questioned signature from `data/samples/`
2. Extract feature embeddings with the SigNet encoder
3. Compute cosine similarity score
4. Display: genuine / forged verdict + confidence score

**Forensic use cases:**
- Verification of signatures on bank cheques
- Authentication of signatures on contracts, wills, and legal deeds
- Detection of traced or digitally reproduced signatures

**Prerequisites:** `torch`, `scikit-image`, `Pillow`

**Note:** Pre-trained SigNet weights (`models/signet.pth`) were trained on the **GPDS** signature dataset. Demo sample pairs are sourced from the **CEDAR** signature database (`data/samples/genuine_N_M.png` / `forged_N_M.png`) and pre-selected so the model correctly detects the forgery.

**Reference:** [luizgh/sigver](https://github.com/luizgh/sigver) — SigNet implementation and pre-trained weights.

---

## Lab 04 — Signature Detection in Documents (YOLOv8)

**File:** `notebooks/04_signature_detection_yolo.ipynb`

Uses a **YOLOv8** model fine-tuned for signature detection (`tech4humans/yolov8s-signature-detector` on Hugging Face) to automatically locate signatures within scanned documents.

**What you will learn:**
- How YOLO object detection works in the context of document analysis
- How to load a Hugging Face model with the `ultralytics` library
- How to run inference on document images and parse bounding box results
- How to visualise detected regions and crop them for downstream processing

**Demo flow:**
1. Load a scanned document image from `data/samples/`
2. Run YOLOv8 inference
3. Draw bounding boxes around detected signatures
4. Crop and save each detected signature for use in Lab 03

**Forensic use cases:**
- Automated extraction of signatures from multi-page legal documents
- First step in a pipeline: detect → extract → verify
- Screening large document archives for signature presence

**Prerequisites:** `ultralytics`, `opencv-python`, `Pillow`

---

## Lab 05 — Writer Identification

**File:** `notebooks/05_writer_identification.ipynb`

Compares the stylistic characteristics of an anonymous handwritten sample against a set of known reference samples to attribute authorship.

**What you will learn:**
- How to extract handwriting style features: HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns), and horizontal/vertical run-length statistics
- How to build an SVM-based writer identification pipeline with scikit-learn (`StandardScaler` + `SVC` with RBF kernel)
- How to evaluate identification accuracy using cross-validation
- How to present results as a ranked list of candidate authors with probability scores

**Demo flow:**
1. Load the reference sample database from `data/samples/writer_XX/` (five writers, 41 samples each)
2. Extract HOG + LBP + run-length features from each sample
3. Train an SVM classifier (`C=10`, `gamma="scale"`, `probability=True`)
4. Upload an anonymous sample → ranked candidate list with probability scores

**Forensic use cases:**
- Attribution of anonymous threatening letters
- Verification of disputed document authorship
- Historical document provenance research

**Prerequisites:** `scikit-learn`, `scikit-image`, `Pillow`, `numpy`

**Dataset note:** The demo uses a synthetic handwriting database in `data/samples/writer_XX/` (five writers, 41 samples each) generated with system TTF fonts (Ink Free, Lucida Handwriting, Segoe Print, Segoe Script, Comic Sans) to ensure distinct, reproducible styles. For production use, replace these with real handwriting scans. The [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) is the standard forensic benchmark.

---

## Lab 06 — Graphological Feature Analysis

**File:** `notebooks/06_graphological_feature_analysis.ipynb`

Automatically extracts and visualises graphological features from a handwritten text sample using classical computer vision and signal processing.

**What you will learn:**
- How to segment handwriting into words and characters using OpenCV
- How to measure: letter slant angle, word/character spacing, letter height and width, stroke pressure (pixel intensity distribution)
- How to build a visual dashboard of graphological metrics
- How to compare two samples and highlight differences

**Demo flow:**
1. Load a handwritten text image from `data/samples/`
2. Preprocess (binarise, denoise, deskew)
3. Segment into lines, words, and characters
4. Compute and display metrics with annotated visualisations
5. (Optional) Compare two samples side-by-side

**Forensic use cases:**
- Supporting expert witness testimony with objective, reproducible measurements
- Detecting stress indicators in handwriting (pen pressure variation)
- Comparative analysis between a reference sample and a disputed document

**Prerequisites:** `opencv-python`, `numpy`, `matplotlib`, `scipy`

---

## Lab 07 — Named Entity Recognition (NER)

**File:** `notebooks/07_named_entity_recognition.ipynb`

Uses a multilingual **BERT-NER** model (`Babelscape/wikineural-multilingual-ner`) to automatically extract named entities — persons, organisations, and locations — from text. Ideal as a second step after HTR transcription.

**What you will learn:**
- How BERT-based token classification works (BIO tagging scheme)
- How to load and run a multilingual NER pipeline via `transformers`
- How to visualise entity spans with colour-coded inline highlighting
- How to build a complete HTR → NER pipeline for handwritten document analysis

**Demo flow:**
1. Run NER on an Italian legal text (e.g. a will or declaration)
2. Run NER on an English text (multilingual support)
3. Full pipeline: load a handwritten image → TrOCR transcription → NER entity extraction
4. Analyse entity distribution and confidence scores

**Forensic use cases:**
- Automatically identify persons, places, and organisations mentioned in handwritten documents
- Screen anonymous letters for proper nouns (names, addresses)
- Build a relationship graph between entities across a document corpus

**Prerequisites:** `transformers`, `torch`, `opencv-python`, `Pillow`, `matplotlib`

---

## Lab 08 — dots.ocr: OCR with Vision-Language Model

**File:** `notebooks/08_dots_ocr_vlm.ipynb`

Uses **dots.ocr** (`rednote-hilab/dots.ocr`) — a 1.7B Vision-Language Model — to transcribe handwritten
text from document images. Unlike CNN-based OCR, the LLM component uses linguistic context to correct
visual ambiguities, making it particularly effective on Italian cursive.

**What you will learn:**

- How VLM-based OCR differs from TrOCR and EasyOCR (architecture comparison table)
- How to perform a hardware check and select the right inference configuration (CPU / GPU)
- How to load and run dots.ocr via `transformers` with adaptive dtype and attention
- How to measure and compare transcription quality (CER) between EasyOCR and dots.ocr

**Demo flow:**

1. Hardware check — detects GPU/RAM and selects CPU fp32 or CUDA bf16 automatically
2. Load dots.ocr from Hugging Face (~3.5 GB bf16, ~7 GB fp32)
3. Transcribe a writer_00 sample (single 320×140 image)
4. Transcribe the full `testamento_writer00.png` document
5. Transcribe Lorella real-world handwriting samples
6. Side-by-side EasyOCR vs dots.ocr comparison + CER measurement

**Forensic use cases:**

- High-quality OCR on Italian cursive handwriting
- Documents with tables, formulas, or complex layouts
- When EasyOCR results are insufficient for forensic accuracy requirements

**Prerequisites:** `transformers>=4.49`, `qwen_vl_utils`, `torch`, `Pillow`, `psutil`, `accelerate`

**Hardware note:** On CPU (~7 GB free RAM), inference takes 2–5 min per image. On GPU ≥8 GB VRAM,
5–10 seconds per image. Not suitable for interactive real-time demos — use EasyOCR in the Gradio app.

**Installation (one-time — see notebook cell):**

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git DotsOCR
pip install -e DotsOCR
pip install qwen_vl_utils accelerate
```

**Reference:** [arxiv 2512.02498](https://arxiv.org/abs/2512.02498) — RedNote / Xiaohongshu, Dec 2024.

---

## Interactive Demo (Gradio)

**File:** `app/grapholab_demo.py`

A browser-based multi-tab Gradio application (fully in Italian) aggregating all seven AI capabilities:

| Tab | Functionality |
|-----|--------------|
| OCR Manoscritto | Upload an image (single or multi-line) → transcribed text (EasyOCR) |
| Verifica Firma | Upload two signatures → genuine / forged verdict |
| Rilevamento Firma | Upload a document → annotated image with detected signatures |
| Riconoscimento Entità | Enter text → colour-coded named entities + summary table |
| Identificazione Scrittore | Upload a handwriting sample → ranked candidate authors with probability scores |
| Analisi Grafologica | Upload handwritten text → visual metrics dashboard |
| Pipeline Forense | Upload a document (+optional reference signature) → full forensic report (all 6 steps) |

**Running locally:**

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app/grapholab_demo.py
# Open http://localhost:7860
```

---

## Running with Docker

```bash
# JupyterLab at http://localhost:8888  (token: grapholab)
docker compose up jupyter

# Gradio demo at http://localhost:7860
docker compose up gradio

# Both services together
docker compose up
```

The Hugging Face model cache is stored in a named Docker volume (`grapholab-hf-cache`) and shared between both services. Models are downloaded only once.

---

## Sample Data

Demo images are provided in `data/samples/`:

| File | Used in |
|------|---------|
| `handwritten_text_01.png` | Labs 02, 06, 07 (single-line HTR demo) |
| `handwritten_multiline_01.png` | Labs 02, 07 (multi-line HTR + NER pipeline) |
| `genuine_N_M.png` | Lab 03 (CEDAR reference signatures, N=writer, M=sample) |
| `forged_N_M.png` | Lab 03 (CEDAR forgeries, pre-selected for model detectability) |
| `document_with_signature_*.png` | Lab 04 |

Signature samples (`genuine_*`, `forged_*`) are sourced from the **CEDAR** signature database and pre-selected so that SigNet correctly classifies the forgery. You can replace or supplement these with your own images to experiment with real-world cases.
