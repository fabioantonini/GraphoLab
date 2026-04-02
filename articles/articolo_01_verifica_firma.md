# Can AI Detect a Forged Signature? The Technology Behind Forensic Handwriting Analysis

*Part 1 of the GraphoLab Series — AI for Forensic Document Examination*

---

## The Disputed Will of Howard Hughes

In April 1976, just days after the reclusive billionaire Howard Hughes died aboard his private jet, a handwritten will surfaced in the offices of the Church of Jesus Christ of Latter-day Saints in Salt Lake City.

The document — three pages, written in pencil — left one-sixth of Hughes' multi-billion dollar estate to a gas station attendant from Nevada named Melvin Dummar, who claimed Hughes had once picked him up hitchhiking in the desert.

Three forensic handwriting experts declared the signature authentic. Three others said it was a forgery.

After years of legal battles and millions spent on expert testimony, the will was ultimately rejected. But the case left a disturbing question unanswered: **if trained human experts can reach diametrically opposite conclusions, is there a more objective way to verify a signature?**

`[IMAGE 1: Historical photo of the Hughes contested will or a representative legal document with signature]`

---

## What a Human Expert Actually Does

When a forensic document examiner is asked to verify a signature, they perform a painstaking visual analysis:

- **Stroke continuity:** Is the pen movement fluid, or does it show signs of slow drawing — a hallmark of careful forgery?
- **Pressure patterns:** Does the ink density match the signer's natural hand pressure?
- **Proportions and spacing:** Are letter heights and word spacing consistent with reference samples?
- **Tremor and hesitation marks:** Forgers slow down to be accurate; authentic signers speed up.

This process is subjective by nature. Two experts can look at the same signature and reach opposite conclusions — as the Hughes case demonstrated. Furthermore, even the *same person* signs differently from one instance to the next: research shows natural intra-writer variability of **15–25%** across repeated signatures.

The challenge for AI is not to replace human judgment, but to provide a **quantifiable, reproducible measurement** that experts can use as an objective anchor.

`[IMAGE 2: Side-by-side comparison of a genuine signature pair vs. a reference + forgery pair — use genuine_1_1.png, genuine_1_2.png, forged_1_1.png from data/samples/]`

---

## How AI Approaches the Problem: Siamese Networks and Metric Learning

The breakthrough came from a deceptively simple idea: instead of teaching a neural network *what a signature looks like*, teach it to **measure the distance between two signatures**.

This approach — called **metric learning** — works as follows: the network maps every signature to a point in a high-dimensional mathematical space. Genuine pairs from the same person cluster close together; forgeries end up far away.

Think of it like a wine sommelier who doesn't memorize every bottle, but learns to recognize subtle differences in taste, aroma, and structure. The sommelier doesn't know what wine "looks like" — they know how to *compare*.

`[IMAGE 3: Simplified diagram — two signature images feed into two network branches, producing vectors, with a distance measurement between them. Create this as a clean infographic.]`

### The SigNet Architecture

GraphoLab uses **SigNet**, a convolutional neural network proposed by Hafemann, Sabourin and Oliveira in their landmark 2017 paper *"Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks"* (Pattern Recognition, Elsevier).

The architecture processes each signature as a grayscale image (150×220 pixels) through:

- **5 convolutional layers** that extract progressively abstract visual features (96 → 256 → 384 → 384 → 256 filters)
- **2 fully connected layers** of 2,048 units each
- A final **L2 normalization** step that projects every signature onto the surface of a unit hypersphere

The result: a **2,048-dimensional vector** — a unique numerical fingerprint for every signature.

### The Verification Decision

Once both signatures are encoded, GraphoLab computes the **cosine distance** between their vectors:

- Distance **< 0.35** → **AUTHENTIC ✓**
- Distance **≥ 0.35** → **FORGED ✗**

When two reference signatures are available, the system averages their embeddings before comparison — improving robustness against natural intra-writer variability.

The model's weights were pre-trained on the **GPDS dataset** (4,000 subjects, 24 genuine + 30 forged signatures each), via the open-source [luizgh/sigver](https://github.com/luizgh/sigver) repository.

---

## GraphoLab in Action

Here is what the verification looks like in practice, using the sample signatures included in the project.

`[IMAGE 4: Screenshot of the GraphoLab "Verifica Firma" tab — reference signature loaded, genuine query, result: AUTHENTIC ✓ with cosine similarity ~0.97]`

`[IMAGE 5: Same tab — forged query loaded — result: FORGED ✗ with cosine distance > 0.35 and the visual comparison gauge]`

The notebook `03_signature_verification_siamese.ipynb` walks through the full pipeline step by step: preprocessing, embedding extraction, distance computation, and — most visually striking — a distribution plot showing how genuine pairs and forgeries naturally separate in similarity space.

`[IMAGE 6: Screenshot from notebook cells 18-19 — histogram showing genuine pairs (high similarity) vs. forged pairs (low similarity) with the decision threshold line]`

This distribution chart is one of the clearest ways to understand why the threshold works: the two populations overlap very little, and the gap between them is where the decision boundary sits.

---

## The State of the Art: Who Else Is Working on This?

Signature verification is an active research area with significant commercial stakes — banks alone lose billions annually to check and payment fraud.

**On the research side:**
- **SigNet-S** (2022): a compact variant designed for edge devices and real-time verification
- **WD-CAPSNET** (2021): capsule networks that model spatial relationships between signature strokes
- **DeepSignDB** (2021): a multimodal benchmark combining offline (image) and online (pen trajectory) data

**In commercial deployments:**

| System | Company | Primary Use |
|---|---|---|
| Topaz Signa | Topaz Systems | Tablet-based signing, banking |
| DynaSign | Softpro | European banks, notaries |
| BBVA AI Signature Fraud | BBVA | Real-time payment fraud detection |
| Adobe Acrobat AI | Adobe | Detecting alterations in signed PDFs |

**In forensic institutions:**
- The **CEDAR** lab (SUNY Buffalo) has been a key contributor to signature analysis research and provides tools used by law enforcement agencies
- The **FISH** (Forensic Information System for Handwriting) database supports handwriting comparison across European forensic labs

---

## Honest Limitations — What AI Cannot Do Yet

Transparency matters. Here is what the current approach gets wrong, or simply cannot handle:

- **The threshold is global, not personal.** A cosine distance of 0.35 works well on average, but every person's natural variability is different. A robust production system would calibrate per individual.
- **Degraded signatures look forged.** Illness, age, stress, or signing quickly on a touchscreen can push a genuine signature well beyond the threshold.
- **Training data bias.** SigNet was trained primarily on Western handwriting conventions. Performance on Arabic, Chinese, Indic, or other script systems is significantly lower.
- **Offline only.** The system works on signature *images*. Online verification — using pen speed and pressure in real time — is a separate and complementary problem.
- **Not admissible as standalone evidence.** In Italian and European legal proceedings, AI-assisted analysis must be validated and interpreted by a qualified court-appointed expert. GraphoLab is a demonstrative and educational tool — it is a starting point for investigation, not a verdict.

> AI provides a measurement. The interpretation remains a human responsibility.

---

## Try It Yourself

GraphoLab is publicly available — no account required, no installation needed:

🔗 **[grapholab on Hugging Face Spaces](https://huggingface.co/spaces/fabioantonini/grapholab)**

Load a reference signature, upload a second image, and see the result in seconds. The full source code and Jupyter notebooks are on GitHub.

---

*Next in the series: **How AI Finds Signatures in Documents** — using Conditional DETR, a Transformer-based object detection architecture, to locate and extract signatures from contracts, wills, and legal records.*

---

*Fabrizio Antonini — AI Engineer & Researcher | GraphoLab Project*

---

### References

- Hafemann, L. G., Sabourin, R., & Oliveira, L. S. (2017). *Learning features for offline handwritten signature verification using deep convolutional neural networks.* Pattern Recognition, 70, 163–176.
- GPDS Corpus Signature dataset — Universidad de Las Palmas de Gran Canaria
- luizgh/sigver — open-source SigNet implementation and pre-trained weights: https://github.com/luizgh/sigver
- CEDAR Signature Database — SUNY Buffalo Center of Excellence for Document Analysis and Recognition

---

`[SERIES BANNER: GraphoLab series cover image — Article 1 of 10]`
