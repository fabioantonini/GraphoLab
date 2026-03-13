# GraphoLab — Demo Checklist

Everything you need to run all six GraphoLab notebooks end-to-end, including which AI models are downloaded automatically and which sample images you must provide.

---

## TL;DR

| What | Where | Notes |
|------|-------|-------|
| Python environment | local or Docker | see [NOTEBOOKS_GUIDE.md](NOTEBOOKS_GUIDE.md) |
| `requirements.txt` installed | — | `pip install -r requirements.txt` |
| Sample images | `data/samples/` | see per-lab sections below |
| AI models | downloaded automatically | internet connection needed on first run |

---

## AI Models — Downloaded Automatically

No manual weight downloads are required. All models are fetched on first run and cached locally (or in the Docker named volume `grapholab-hf-cache`).

| Model | Downloaded by | Size | Cache location |
|-------|--------------|------|----------------|
| **ResNet-18** (ImageNet weights) | `torchvision` | ~45 MB | `~/.cache/torch/hub/` |
| **TrOCR** (`microsoft/trocr-base-handwritten`) | `transformers` | ~400 MB | `~/.cache/huggingface/` |
| **YOLOv8s signature detector** (`tech4humans/yolov8s-signature-detector`) | `huggingface_hub` | ~22 MB | `~/.cache/huggingface/` |

> **Internet connection is required on the first run of Labs 02, 03, and 04.** Subsequent runs use the cached models.

---

## Sample Images — What You Need to Provide

Place all images in `data/samples/`. Synthetic placeholder images are generated automatically when real images are missing, so the notebooks always run — but results on synthetic data are not meaningful for real forensic use.

### Lab 01 — Introduction
**Nothing required.** Markdown-only notebook.

---

### Lab 02 — Handwritten Text Recognition (TrOCR)

| File | Description |
|------|-------------|
| `handwritten_text_01.png` | A single line of handwritten text |
| `handwritten_text_02.png` | (optional) A second sample for batch demo |

**Requirements:**
- Clear scan or photo of handwritten text
- Recommended resolution: 300 DPI or higher
- White or light background, dark ink
- Single line or a short paragraph works best

**Ground-truth comparison (optional):** if you have a known transcript of the handwritten text, you can compute the Character Error Rate (CER) in the optional section of Lab 02.

---

### Lab 03 — Signature Verification (ResNet-18)

| File | Description |
|------|-------------|
| `signature_genuine_01.png` | **Reference signature** — known genuine |
| `signature_genuine_02.png` | Second genuine signature from the same person |
| `signature_forged_01.png` | A forged or questioned signature |

For the **distribution plot** (Demo 5), add more files following the same naming pattern:
- `signature_genuine_03.png`, `signature_genuine_04.png`, … (additional genuine samples)
- `signature_forged_02.png`, `signature_forged_03.png`, … (additional forged samples)

**Requirements:**
- Isolated signatures (no surrounding document text)
- White or light background, dark ink
- Consistent scan quality across samples from the same person
- Recommended resolution: 300 DPI or higher

> **Minimum for a meaningful demo:** 1 reference + 1 genuine copy + 1 forged. The model (ResNet-18 ImageNet weights) works out of the box — no training required.

---

### Lab 04 — Signature Detection in Documents (YOLOv8)

| File | Description |
|------|-------------|
| `document_with_signature_01.png` | A scanned document page containing at least one signature |

**Optional additional files:** `document_with_signature_02.png`, `document_with_signature_03.png`, …

**Requirements:**
- Full document page image (not a pre-cropped signature)
- The model handles multi-signature pages
- Recommended resolution: 200–300 DPI
- Works on contracts, letters, forms, bank cheques

> **Output:** detected signatures are cropped and saved as `detected_signature_N.png` in `data/samples/`. These crops can be used directly as input to Lab 03.

---

### Lab 05 — Writer Identification

Organised in per-writer subdirectories inside `data/samples/`:

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

**Requirements:**
- Minimum **3 writers** (more = better accuracy)
- Minimum **5 samples per writer** (the notebook uses leave-one-out cross-validation)
- Each sample: a few lines of continuous handwritten text
- Consistent scan conditions across all samples
- Recommended resolution: 300 DPI

> **Training note:** Lab 05 trains a lightweight SVM classifier on the provided samples each time the notebook runs. No pre-trained writer identification model is used — your own samples are the training data.

---

### Lab 06 — Graphological Feature Analysis

Reuses the handwritten text images from Lab 02:

| File | Description |
|------|-------------|
| `handwritten_text_01.png` | Primary sample for feature extraction |
| `handwritten_text_02.png` | (optional) Second sample for side-by-side comparison |

No additional files needed if Lab 02 samples are already in place.

---

## Naming Convention Summary

```
data/samples/
  handwritten_text_01.png          # Labs 02, 06
  handwritten_text_02.png          # Labs 02, 06 (optional)
  signature_genuine_01.png         # Lab 03 — reference
  signature_genuine_02.png         # Lab 03 — genuine copy
  signature_genuine_03.png         # Lab 03 — distribution plot (optional)
  signature_forged_01.png          # Lab 03 — forged / questioned
  signature_forged_02.png          # Lab 03 — distribution plot (optional)
  document_with_signature_01.png   # Lab 04
  writer_01/sample_01.png          # Lab 05
  writer_01/sample_02.png          # Lab 05
  ...
```

---

## Minimum Viable Demo (5 images)

If you want a quick demo covering Labs 02, 03, 04, and 06 with a single minimal set:

1. `handwritten_text_01.png` — for Labs 02 and 06
2. `signature_genuine_01.png` — reference signature
3. `signature_genuine_02.png` — second genuine signature
4. `signature_forged_01.png` — forged signature
5. `document_with_signature_01.png` — document page for Lab 04

Lab 01 needs nothing. Lab 05 needs per-writer subdirectories (not covered by this minimum set).

---

## Quick Checklist Before Running

- [ ] Python environment created and `requirements.txt` installed
- [ ] Internet connection available (first-run model downloads)
- [ ] `data/samples/` directory exists
- [ ] Handwritten text images placed (`handwritten_text_*.png`)
- [ ] Signature images placed (`signature_genuine_*.png`, `signature_forged_*.png`)
- [ ] Document scan placed (`document_with_signature_*.png`)
- [ ] Writer subdirectories populated (`writer_XX/sample_YY.png`) — for Lab 05
- [ ] JupyterLab running (`jupyter lab` or `docker compose up jupyter`)
