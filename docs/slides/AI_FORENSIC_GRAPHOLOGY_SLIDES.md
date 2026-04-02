---
marp: true
theme: gaia
class: invert
paginate: true
footer: "GraphoLab — AI in Forensic Graphology"
style: |
  section {
    font-size: 1.6rem;
  }
  section.lead h1 {
    font-size: 2.4rem;
  }
  section.lead h2 {
    font-size: 1.6rem;
    font-weight: normal;
    opacity: 0.85;
  }
  h2 {
    color: #a8d8f0;
    border-bottom: 2px solid #a8d8f0;
    padding-bottom: 0.2em;
  }
  strong {
    color: #ffd166;
  }
  table {
    font-size: 1.3rem;
  }
  section.divider {
    justify-content: center;
    text-align: center;
    background: #1a1a2e;
  }
  section.divider h1 {
    font-size: 2.2rem;
    color: #a8d8f0;
  }
  section.divider p {
    opacity: 0.7;
    font-size: 1.3rem;
  }
---

<!-- _class: lead -->

# Artificial Intelligence and<br>Forensic Graphology

### How machine learning is transforming the analysis<br>of handwriting and signatures

<br>

**GraphoLab** — Demo Laboratory

---

## Agenda

1. What is forensic graphology?
2. Limits of traditional manual analysis
3. Why AI is a natural complement
4. Five areas of application
5. AI as a support tool — not a replacement
6. Ethical and legal considerations
7. The GraphoLab laboratories

---

<!-- _class: divider -->

# Part 1
## The Domain

---

## What is Forensic Graphology?

> *The scientific examination of handwriting and signatures to answer questions relevant to legal proceedings.*

Typical questions in court:

- Is this **signature genuine**, or has it been **forged**?
- **Who wrote** this anonymous letter?
- Has this **document been altered** after signing?
- Is the handwriting **consistent** with the known author?

---

## The Traditional Workflow

A forensic document examiner follows these steps:

| Step | Activity |
|------|----------|
| **Acquisition** | Scan the document at 600–1200 DPI |
| **Reference collection** | Gather known samples from candidate authors |
| **Visual comparison** | Analyse slant, pressure, spacing, letter forms |
| **Expert opinion** | Write a report for court |

---

<!-- _class: divider -->

# Part 2
## Limits of Manual Analysis

---

## Where Manual Analysis Falls Short

**Time and scalability**
Comparing dozens of samples or searching hundreds of pages is slow. In large investigations, it becomes a bottleneck.

**Subjectivity**
Borderline cases can yield different opinions from different examiners — not a failing, but an inherent limitation of visual judgement.

**Reproducibility**
Traditional reports are qualitative. A third party cannot easily verify *how* a feature was measured or *how much weight* it was given.

**Scale of modern archives**
Legal disputes increasingly involve thousands of scanned documents. Manual page-by-page review is simply not feasible.

---

<!-- _class: divider -->

# Part 3
## Why AI Is a Natural Complement

---

## AI Adds What Manual Analysis Lacks

| Challenge | AI Contribution |
|-----------|----------------|
| Speed | Thousands of images processed in minutes |
| Objectivity | Precise numerical measurements, not impressions |
| Reproducibility | Same algorithm, same result — every time |
| Scale | Large archives screened automatically |
| Transparency | Every step logged → full audit trail |

---

## How AI "Reads" Handwriting

AI uses **computer vision** — teaching machines to recognise patterns in images.

Applied to handwriting, this means:

- Detecting **subtle regularities** invisible to the naked eye
- Measuring features **precisely and consistently**
- Comparing samples **statistically**, not just visually
- Producing outputs that are **quantifiable and explainable**

> Handwriting is rich in patterns. AI is very good at patterns.

---

<!-- _class: divider -->

# Part 4
## Five Areas of Application

---

## 1. Handwritten Text Recognition (HTR)

**What it does**
Converts handwriting images directly into machine-readable text — like a very sophisticated typist.

**Why it matters forensically**
- Anonymous letters → instant transcription and keyword search
- Historical court documents → digitised and searchable
- Long documents → processed in seconds, not hours

**What it does NOT do**
It does not interpret meaning or attribute authorship. It converts image to words.

> *Lab 02 in GraphoLab*

---

## 2. Signature Authenticity Verification

**What it does**
Compares a questioned signature against known genuine reference samples and returns a **similarity score** with a confidence level.

**Why it matters forensically**
- Bank cheques, contracts, wills, legal deeds
- Detects traced or digitally reproduced signatures
- Quantifies how "far" a questioned signature is from the reference

**What it does NOT do**
It does not deliver a legal verdict. It provides a quantitative input for the expert.

> *Lab 03 in GraphoLab*

---

## 3. Signature Detection in Documents

**What it does**
Automatically scans a document image and **locates all signatures**, drawing a bounding box around each one.

**Why it matters forensically**
- Multi-page contracts → signatures found instantly
- Large archives → screened in bulk
- Feeds directly into the verification pipeline: **detect → extract → verify**

**What it does NOT do**
It does not distinguish genuine from forged — that is the next step.

> *Lab 04 in GraphoLab*

---

## 4. Writer Identification

**What it does**
Analyses an anonymous handwriting sample and returns a **ranked list of candidate authors** with probability scores, comparing against a reference database.

**Why it matters forensically**
- Anonymous threatening letters → candidate authors identified
- Disputed document authorship → statistical evidence of attribution
- Historical manuscripts → provenance research

**What it does NOT do**
A high probability score is not proof of authorship. It is a statistical indication for further expert examination.

> *Lab 05 in GraphoLab*

---

## 5. Graphological Feature Analysis

**What it does**
Automatically extracts and quantifies graphological characteristics from a handwriting image.

| Feature | What it reveals |
|---------|----------------|
| **Letter slant** | Habitual style; disguise attempts |
| **Stroke pressure** | Pen type, physical state |
| **Letter size** | Consistency across samples |
| **Word spacing** | Writing speed and habits |
| **Baseline regularity** | Fatigue, stress indicators |

> *Lab 06 in GraphoLab*

---

<!-- _class: divider -->

# Part 5
## AI as a Support Tool

---

## Human + AI Collaboration

AI is not a replacement for the forensic expert. It is a **more powerful instrument** in their hands.

```
AI handles:                    Expert handles:
─────────────────              ──────────────────────────
• Processing scale             • Contextual interpretation
• Numerical measurements       • Professional judgement
• Consistency                  • Legal accountability
• Audit trail                  • Court testimony
```

**The model:** AI compresses the quantitative work so the expert can focus on what only humans can do.

---

## What AI Outputs Are — and Aren't

| AI Produces | AI Does NOT Produce |
|-------------|-------------------|
| Similarity **scores** | Legal **verdicts** |
| Probability **rankings** | Proof of authorship |
| Numerical **measurements** | Conclusive expert opinion |
| Statistical **indicators** | Contextual judgement |

> All AI outputs require validation by a qualified forensic document examiner before use in court.

---

<!-- _class: divider -->

# Part 6
## Ethical and Legal Considerations

---

## Key Considerations

**Bias in training data**
Models trained on limited populations may perform less reliably on under-represented handwriting styles or languages. Always understand the model's training data.

**Explainability**
Courts require justifiable conclusions. AI systems must be used in ways that can be explained clearly to judges, juries, and opposing counsel.

**Chain of custody**
Digital evidence must be handled rigorously. Every step of AI-assisted analysis must be documented; original files must remain unaltered.

**Reference standards**
OSAC and SWGDOC publish best-practice frameworks for forensic document examination, including computational tools.

---

<!-- _class: divider -->

# Part 7
## The GraphoLab Laboratories

---

## GraphoLab — Lab Overview

| Lab | Title | Core Technology |
|-----|-------|----------------|
| 01 | Introduction | Conceptual overview |
| 02 | Handwritten OCR | TrOCR (Transformer) |
| 03 | Signature Verification | Siamese Neural Network |
| 04 | Signature Detection | Conditional DETR Object Detection |
| 05 | Writer Identification | Feature extraction + SVM |
| 06 | Graphological Analysis | OpenCV + Image Processing |
| — | Gradio Demo App | All capabilities, browser-based |

Each lab includes working code, synthetic fallback data, and a forensic use case section.

---

## Running the Labs

**Local (Python + JupyterLab)**
```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

**Docker — JupyterLab**
```bash
docker compose up jupyter
# http://localhost:8888  (token: grapholab)
```

**Docker — Interactive Demo**
```bash
docker compose up gradio
# http://localhost:7860
```

---

<!-- _class: lead -->

# Conclusion

AI does not replace the forensic expert.

It makes their work **faster**, **more objective**, and **more reproducible**.

<br>

The goal is not a machine that delivers verdicts —
but a more powerful instrument in the hands
of the qualified professional.

<br>

**GraphoLab** · [github.com/fabioantonini/GraphoLab](https://github.com/fabioantonini/GraphoLab)
