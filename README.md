# GraphoLab — Forensic Graphology Laboratory

A collection of AI-powered demo labs showing how machine learning and computer vision can be applied to **forensic graphology**: the scientific examination of handwriting and signatures for legal purposes.

---

## Labs Overview

| Lab | Notebook | AI Technique | Model / Tool |
|-----|----------|-------------|--------------|
| 01 | [Introduction](notebooks/01_intro_forensic_graphology.ipynb) | — | Conceptual overview |
| 02 | [Handwritten OCR](notebooks/02_handwritten_ocr_trocr.ipynb) | Transformer OCR | `microsoft/trocr-base-handwritten` |
| 03 | [Signature Verification](notebooks/03_signature_verification_siamese.ipynb) | Siamese Network | SigNet (luizgh/sigver) |
| 04 | [Signature Detection](notebooks/04_signature_detection_yolo.ipynb) | Object Detection | YOLOv8 (tech4humans) |
| 05 | [Writer Identification](notebooks/05_writer_identification.ipynb) | HOG + SVM | scikit-learn |
| 06 | [Graphological Analysis](notebooks/06_graphological_feature_analysis.ipynb) | Image Processing | OpenCV |

See [docs/NOTEBOOKS_GUIDE.md](docs/NOTEBOOKS_GUIDE.md) for a full description of each lab.

---

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

### Gradio Interactive Demo

```bash
pip install -r requirements.txt
python app/grapholab_demo.py
# Open http://localhost:7860
```

### Docker

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
├── notebooks/              ← Jupyter labs (01–06)
├── app/
│   └── grapholab_demo.py   ← Gradio interactive demo
├── data/
│   └── samples/            ← Place your handwriting/signature images here
├── docs/
│   └── NOTEBOOKS_GUIDE.md  ← Detailed lab descriptions
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Sample Data

Add handwriting and signature images to `data/samples/`. Each notebook also generates synthetic fallback images automatically, so labs run end-to-end without any real data.

See [data/samples/README.md](data/samples/README.md) for naming conventions.

---

## Key Models & Resources

| Use case | Resource |
|----------|----------|
| Handwritten OCR | [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) |
| Signature Detection | [tech4humans/yolov8s-signature-detector](https://huggingface.co/tech4humans/yolov8s-signature-detector) |
| Signature Verification | [luizgh/sigver](https://github.com/luizgh/sigver) |
| Graphology ML reference | [CVxTz/handwriting_forensics](https://github.com/CVxTz/handwriting_forensics) |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
