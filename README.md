---
title: GraphoLab — AI for Forensic Graphology
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app/grapholab_demo.py
pinned: false
license: apache-2.0
short_description: AI-powered forensic graphology demo (HTR, signatures, NER)
---

# GraphoLab — Forensic Graphology Laboratory

A collection of AI-powered demo labs showing how machine learning and computer vision can be applied to **forensic graphology**: the scientific examination of handwriting and signatures for legal purposes.

---

## Labs Overview

### Jupyter Notebooks (offline)

| Lab | Notebook | AI Technique | Model / Tool |
|-----|----------|-------------|--------------|
| 01 | [Introduction](notebooks/01_intro_forensic_graphology.ipynb) | — | Conceptual overview |
| 02 | [Handwritten OCR](notebooks/02_handwritten_ocr_trocr.ipynb) | Transformer OCR | `microsoft/trocr-base-handwritten` |
| 03 | [Signature Verification](notebooks/03_signature_verification_siamese.ipynb) | Siamese Network | SigNet (luizgh/sigver) |
| 04 | [Signature Detection](notebooks/04_signature_detection_yolo.ipynb) | Object Detection | YOLOv8 (tech4humans) |
| 05 | [Writer Identification](notebooks/05_writer_identification.ipynb) | HOG + SVM | scikit-learn |
| 06 | [Graphological Analysis](notebooks/06_graphological_feature_analysis.ipynb) | Image Processing | OpenCV |
| 07 | [Named Entity Recognition](notebooks/07_named_entity_recognition.ipynb) | Token Classification | `Babelscape/wikineural-multilingual-ner` |
| 08 | [dots.ocr VLM](notebooks/08_dots_ocr_vlm.ipynb) | Vision-Language Model | `rednote-hilab/dots.ocr` (1.7B params) |

See [docs/NOTEBOOKS_GUIDE.md](docs/NOTEBOOKS_GUIDE.md) for a full description of each lab.

### Gradio Demo Tabs (interactive)

| Tab | Name | Description |
|-----|------|-------------|
| 1 | OCR Manoscritto | Handwritten text transcription with EasyOCR |
| 2 | Verifica Firma | Signature verification with SigNet (Siamese network) |
| 3 | Rilevamento Firma | Signature detection in documents with YOLOv8 |
| 4 | Riconoscimento Entità | Named entity recognition (NER) on transcribed text |
| 5 | Identificazione Scrittore | Writer identification with HOG + SVM |
| 6 | Analisi Grafologica | Graphological feature analysis (slant, pressure, spacing) |
| 7 | Perizia Forense Automatica | **Full forensic pipeline**: runs all 6 AI tools in sequence, then synthesizes a complete forensic report via **Ollama LLM** (local, no data sent online) |
| 8 | Datazione Documenti | Chronological ordering of multiple documents by extracted dates (EasyOCR + dateparser) |
| 9 | Consulente Forense IA | **RAG chatbot**: upload PDF/DOCX documents to enrich the knowledge base, then ask questions in Italian answered by a local Ollama LLM |

---

## Quick Start

### Local (Python)

Requires Python 3.11, 3.12, or 3.13.

```bash
# Create a virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install PyTorch with CUDA 12.4 (GPU) — skip --index-url for CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Run JupyterLab
jupyter lab notebooks/

# Run Gradio demo
python app/grapholab_demo.py
# Open http://localhost:7860
```

### Docker

Docker images use `nvidia/cuda:12.4.1` as base and support GPU automatically via the WSL2 backend on Windows (no extra configuration needed if NVIDIA drivers are installed).

```bash
# JupyterLab at http://localhost:8888  (token: grapholab)
docker compose up jupyter

# Gradio demo at http://localhost:7860
docker compose up gradio

# Both services
docker compose up
```

---

## Project Structure

```
GraphoLab/
├── notebooks/              ← Jupyter labs (01–07)
├── app/
│   └── grapholab_demo.py   ← Gradio interactive demo (9 tabs, Italian UI)
├── data/
│   └── samples/
│       ├── writer_XX/      ← Writer identification database (5 writers, 41 samples each)
│       └── *.png           ← Handwriting and signature sample images
├── docs/
│   ├── NOTEBOOKS_GUIDE.md       ← Detailed lab descriptions (EN)
│   ├── NOTEBOOKS_GUIDE_IT.md    ← Detailed lab descriptions (IT)
│   └── GraphoLab_Presentazione.pptx  ← Presentation with speaker notes
├── models/                 ← Pre-trained model weights (e.g. signet.pth)
├── tools/
│   └── generate_presentation.py ← Script to regenerate the PPTX
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Sample Data

Add handwriting and signature images to `data/samples/`. Each notebook also generates synthetic fallback images automatically, so labs run end-to-end without any real data.

The writer identification database (`data/samples/writer_XX/`) contains five writers with 41 samples each, generated with system TTF fonts (Ink Free, Lucida Handwriting, Segoe Print, Segoe Script, Comic Sans). Replace with real handwriting scans for production use.

See [data/samples/README.md](data/samples/README.md) for naming conventions.

---

## Key Models & Resources

| Use case | Resource |
|----------|----------|
| Handwritten OCR | [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) |
| Handwritten OCR (VLM, notebook only) | [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) — 1.7B Vision-Language Model |
| Signature Detection | [tech4humans/yolov8s-signature-detector](https://huggingface.co/tech4humans/yolov8s-signature-detector) ⚠️ gated |
| Signature Verification | [luizgh/sigver](https://github.com/luizgh/sigver) |
| NER | [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) |
| Forensic report synthesis & RAG chatbot | [Ollama](https://ollama.com) — local LLM (no data sent online) |
| Graphology ML reference | [CVxTz/handwriting_forensics](https://github.com/CVxTz/handwriting_forensics) |

### Ollama — Local LLM (tabs 7 and 9)

Tabs **Perizia Forense Automatica** (7) and **Consulente Forense IA** (9) require a running [Ollama](https://ollama.com) instance with at least one model pulled.

```bash
# Install Ollama, then pull a model (e.g. llama3 or mistral)
ollama pull llama3

# Ollama listens on http://localhost:11434 by default
ollama serve
```

Without Ollama, both tabs degrade gracefully: tab 7 skips the LLM synthesis step and tab 9 shows an informational message.

---

### Signature Detection — Gated Model Access

The `tech4humans/yolov8s-signature-detector` model requires authentication on Hugging Face.

**Steps to enable the Signature Detection tab:**

1. Create or log in to your account at [huggingface.co](https://huggingface.co)
2. Request access at [huggingface.co/tech4humans/yolov8s-signature-detector](https://huggingface.co/tech4humans/yolov8s-signature-detector)
3. Once approved, create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (type: Read)
4. Set the token before starting the app:

```powershell
# Windows PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
venv\Scripts\python app\grapholab_demo.py
```

```bash
# Linux/macOS or Windows bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxx python app/grapholab_demo.py
```

Without the token, the Signature Detection tab will display a friendly error message instead of crashing.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
