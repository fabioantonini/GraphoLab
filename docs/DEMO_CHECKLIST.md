# GraphoLab — Demo Checklist

Everything you need to run all seven GraphoLab notebooks end-to-end, including which AI models are downloaded automatically and which sample images you must provide.

---

## TL;DR

| What | Where | Notes |
|------|-------|-------|
| Python environment | local or Docker | see [NOTEBOOKS_GUIDE.md](NOTEBOOKS_GUIDE.md) |
| `requirements.txt` installed | — | `pip install -r requirements.txt` |
| SigNet weights | `models/signet.pth` | manual download — see Lab 03 section |
| Sample images | `data/samples/` | see per-lab sections below |
| AI models | downloaded automatically | internet connection needed on first run |

---

## AI Models — Downloaded Automatically

All Hugging Face models are fetched on first run and cached locally (or in the Docker named volume `grapholab-hf-cache`).

| Model | Downloaded by | Size | Cache location |
|-------|--------------|------|----------------|
| **TrOCR** (`microsoft/trocr-base-handwritten`) | `transformers` | ~400 MB | `~/.cache/huggingface/` |
| **YOLOv8s signature detector** (`tech4humans/yolov8s-signature-detector`) | `huggingface_hub` | ~22 MB | `~/.cache/huggingface/` |
| **WikiNEural NER** (`Babelscape/wikineural-multilingual-ner`) | `transformers` | ~700 MB | `~/.cache/huggingface/` |

> **Internet connection is required on the first run of Labs 02, 04, and 07.** Subsequent runs use the cached models.

## AI Models — Manual Download Required

| Model | File | Size | Source |
|-------|------|------|--------|
| **SigNet** (GPDS pre-trained) | `models/signet.pth` | ~63 MB | [luizgh/sigver](https://github.com/luizgh/sigver) |

Download `signet.pth` from the sigver repository and place it in the `models/` directory before running Lab 03.

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
| `handwritten_text_02.png` | (optional) A second single-line sample |
| `handwritten_multiline_01.png` | A multi-line handwritten document (for the HTR→NER pipeline demo) |

**Requirements:**
- Clear scan or photo of handwritten text
- Recommended resolution: 300 DPI or higher
- White or light background, dark ink
- TrOCR is a line-level model; multi-line images are split automatically by horizontal projection before inference

**Ground-truth comparison (optional):** if you have a known transcript of the handwritten text, you can compute the Character Error Rate (CER) in the optional section of Lab 02.

---

### Lab 03 — Signature Verification (SigNet)

| File | Description |
|------|-------------|
| `genuine_N_1.png` | **Reference signature** — known genuine (writer N, sample 1) |
| `genuine_N_2.png` | Second genuine signature from the same writer |
| `forged_N_M.png` | A forged signature (writer N, forgery M) |

Repeat for each writer you want to demonstrate (e.g. N = 1, 2, 3, …).

**Requirements:**
- Isolated signatures (no surrounding document text)
- White or light background, dark ink
- Consistent scan quality across samples from the same person
- Recommended resolution: 300 DPI or higher

> **Pre-selected demo samples:** The repository includes curated pairs from the **CEDAR** signature database. These pairs have been pre-scanned with SigNet to confirm the model correctly classifies the forgery (cosine distance > 0.35). Writers 1–5 correspond to CEDAR writers 51, 26, 34, 32, and 21 respectively.

> **SigNet weights required:** download `models/signet.pth` from [luizgh/sigver](https://github.com/luizgh/sigver) before running this lab.

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

### Lab 07 — Named Entity Recognition (NER)

**No image files required.** The NER model operates on text strings directly.

- **Demo 1 & 2:** hard-coded Italian and English example texts — no files needed.
- **Demo 3 (HTR→NER pipeline):** loads `handwritten_multiline_01.png` (shared with Lab 02).

The `Babelscape/wikineural-multilingual-ner` model (~700 MB) is downloaded automatically on first run. It supports 9 languages including Italian and English.

---

## Naming Convention Summary

```
data/samples/
  handwritten_text_01.png          # Labs 02, 06
  handwritten_text_02.png          # Labs 02, 06 (optional)
  handwritten_multiline_01.png     # Labs 02, 07 (multi-line HTR + NER pipeline)
  genuine_1_1.png                  # Lab 03 — writer 1, reference
  genuine_1_2.png                  # Lab 03 — writer 1, second genuine sample
  forged_1_1.png                   # Lab 03 — writer 1, forged
  genuine_2_1.png                  # Lab 03 — writer 2, reference
  ...
  document_with_signature_01.png   # Lab 04
  writer_01/sample_01.png          # Lab 05
  writer_01/sample_02.png          # Lab 05
  ...
```

---

## Minimum Viable Demo (5 images)

If you want a quick demo covering Labs 02, 03, 04, 06, and 07 with a single minimal set:

1. `handwritten_text_01.png` — for Labs 02 and 06
2. `handwritten_multiline_01.png` — for Lab 07 HTR→NER pipeline
3. `genuine_1_1.png` — reference signature
4. `forged_1_1.png` — forged signature
5. `document_with_signature_01.png` — document page for Lab 04

Lab 01 needs nothing. Lab 05 needs per-writer subdirectories (not covered by this minimum set). Lab 07 Demos 1 & 2 need no files at all.

---

## Quick Checklist Before Running

- [ ] Python environment created and `requirements.txt` installed
- [ ] Internet connection available (first-run model downloads: TrOCR ~400 MB, WikiNEural NER ~700 MB, YOLOv8 ~22 MB)
- [ ] `models/signet.pth` downloaded from [luizgh/sigver](https://github.com/luizgh/sigver)
- [ ] `data/samples/` directory exists
- [ ] Handwritten text images placed (`handwritten_text_*.png`, `handwritten_multiline_01.png`)
- [ ] Signature images placed (`genuine_N_M.png`, `forged_N_M.png`)
- [ ] Document scan placed (`document_with_signature_*.png`)
- [ ] Writer subdirectories populated (`writer_XX/sample_YY.png`) — for Lab 05
- [ ] JupyterLab running (`jupyter lab` or `docker compose up jupyter`)
