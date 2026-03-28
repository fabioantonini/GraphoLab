# GraphoLab: When Artificial Intelligence Meets Forensic Graphology

<!--
HOW TO PUBLISH ON LINKEDIN:
1. Open https://markdowntolinkedin.com
2. Paste the content of this file (from the title onwards)
3. Copy the formatted text
4. On LinkedIn → "Write an article" → paste the text
5. Upload docs/linkedin_cover.png as the cover image (1920×1080)
6. Use the H1 title as the LinkedIn article title
-->

---

Picture this: a law firm, a will sitting on the desk, and a forensic graphologist holding a magnifying glass. The question is simple, but the answer can be worth millions — or change the outcome of a criminal trial: **is this signature genuine?**

For decades, answering that question has required years of expertise, hours of painstaking manual work and, inevitably, a degree of subjectivity. Today, artificial intelligence offers new tools to support, accelerate and make this process more rigorous.

I built **GraphoLab** — an open-source collection of eight demonstration labs showing how machine learning and computer vision can be applied to forensic graphology.

---

## The Problem: Limits of Traditional Analysis

Forensic graphology is a serious, well-established discipline. But like any manual process, it has structural limitations:

- **Subjectivity**: two expert examiners can reach different conclusions on the same document
- **Scalability**: finding a specific signature among ten thousand scanned pages takes weeks
- **Reproducibility**: qualitative observations ("the pen pressure seems different") are hard to verify independently
- **Speed**: in complex legal proceedings, the timeline of manual analysis can become a bottleneck

AI does not eliminate these problems — but it significantly reduces them, bringing objective measurements alongside human expertise.

---

## The Solution: GraphoLab

GraphoLab is an open-source project that integrates seven AI technologies into a unified demonstration platform, accessible via browser thanks to a **Gradio** app. Each lab addresses a specific forensic graphology task.

### The Eight Labs

| Lab | Functionality | AI Technology |
|-----|--------------|---------------|
| 01 | Conceptual introduction | — |
| 02 | Handwritten Text Recognition (HTR) | TrOCR / EasyOCR |
| 03 | Signature authenticity verification | SigNet (Siamese Network) |
| 04 | Signature detection in documents | YOLOv8 fine-tuned |
| 05 | Writer identification | HOG + LBP + SVM |
| 06 | Graphological feature analysis | OpenCV + signal processing |
| 07 | Named Entity Recognition (NER) | Multilingual BERT-NER |
| 08 | Advanced OCR for Italian cursive | dots.ocr (1.7B VLM) |

---

## Deep Dive 1: Verifying a Signature Without Knowing the Signer

One of the most frequent questions I get: *"Does SigNet only work with signatures already in its training database?"*

The answer is no — and the reason is architectural.

SigNet uses **metric learning** (a Siamese network with contrastive loss), not a traditional classifier. A classifier learns "who is person X" and cannot generalise to unseen identities. A metric model, by contrast, learns to answer a different question: *"did these two signatures come from the same hand?"*

This question is **identity-agnostic**. SigNet can therefore be applied to any pair of signatures — even from people never seen during training — producing a cosine similarity score that the examiner can use as quantitative evidence.

**Limitations exist and must be communicated clearly:**
- The training set (GPDS) contains primarily Brazilian/Portuguese signatures: styles far from that distribution may yield less reliable scores
- The decision threshold (0.35) was calibrated on CEDAR and may need adjustment in different contexts
- The system is a screening tool, not standalone proof: the final judgement always rests with the qualified examiner

---

## Deep Dive 2: Transcribing Italian Cursive Is an Open Problem

Not all OCR engines are equal — and in the Italian forensic context, the difference can be decisive.

**EasyOCR** (used in the interactive app) runs a CNN + BiLSTM + CTC pipeline: fast (1–3 seconds per image on CPU), but with limited linguistic context. It works well on printed text and regular cursive.

**TrOCR** (Lab 02) is a full Transformer: BEiT visual encoder + RoBERTa decoder. Global linguistic context (self-attention) makes it more accurate on complex cursive, at the cost of 10–20 seconds per image on CPU.

**dots.ocr** (Lab 08) is a 1.7-billion-parameter Vision-Language Model. The LLM component corrects visual ambiguities using sentence-level semantic context — delivering the best publicly available accuracy on Italian cursive, requiring ~7 GB RAM and 2–5 minutes per image on CPU.

The right tool depends on the context: EasyOCR for interactive demos, dots.ocr for forensic transcription of a holographic will.

---

## The Interactive Demo: Eight Tabs, All in a Browser

GraphoLab's Gradio app aggregates all capabilities in a single, accessible interface (no coding required — just Docker):

- **Handwritten OCR** — upload an image, get the transcribed text
- **Signature Verification** — upload two signatures, get a genuine/forged verdict with a confidence score
- **Signature Detection** — upload a multi-page document, automatically extract all signatures
- **Named Entity Recognition** — identify persons, locations and organisations in the text
- **Writer Identification** — attribute authorship of an anonymous handwriting sample
- **Graphological Analysis** — measure slant, spacing and stroke pressure
- **Forensic Pipeline** — full report in a single pass
- **Document Dating** — upload multiple documents, get them sorted chronologically by extracted date

---

## Ethics and Limits: AI Does Not Replace the Expert

This point deserves to be stated plainly.

GraphoLab is a support tool, not an oracle. AI models produce scores and measurements — not verdicts. The appropriate model is **human-AI collaboration**: AI handles the quantitative and labour-intensive aspects of analysis, while the expert focuses on interpretation, contextualisation and legal accountability.

In a courtroom, "the model says X" is not evidence. "The expert used this tool to corroborate their analysis, here is how" is a different matter entirely.

Transparency about model limitations — training datasets, decision thresholds, conditions of validity — is an integral part of responsible use of these tools in forensic practice.

---

## Try It Yourself

GraphoLab is fully open source under the Apache 2.0 licence.

**GitHub repository:** https://github.com/fabioantonini/GraphoLab

You will find:
- Eight runnable Jupyter notebooks (Python 3.11/3.12 + PyTorch)
- The Gradio app, launchable locally or via Docker
- Full documentation in Italian and English
- Scripts to generate synthetic test data

```bash
git clone https://github.com/fabioantonini/GraphoLab.git
cd GraphoLab
docker compose up gradio
# Open http://localhost:7860
```

---

I am curious to hear your thoughts — especially from those working in forensic or legal practice. AI in this domain is still largely unexplored, and I believe there are significant opportunities for professionals who want to bring quantitative rigour to expert witness work.

**Forensic examiners, lawyers, notaries, document specialists: what do you expect from tools like this? What is still missing?**

---

*Fabio Antonini — AI Engineer & Researcher*
*GitHub: https://github.com/fabioantonini/GraphoLab*
