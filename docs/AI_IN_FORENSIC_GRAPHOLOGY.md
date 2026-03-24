# Artificial Intelligence and Forensic Graphology

## Why AI Is Transforming the Analysis of Handwriting and Signatures

---

## Introduction

Every time a person writes by hand or signs a document, they leave behind a trace that is as personal and distinctive as a fingerprint. The way letters are shaped, the angle at which the pen is held, the pressure applied to the paper, the rhythm of the strokes — all of these characteristics form a unique stylistic profile that persists across different samples and over time.

**Forensic graphology** — more formally known as *questioned document examination* — is the scientific discipline that studies these characteristics to answer questions relevant to legal proceedings:

- Is this signature genuine, or has it been forged?
- Who wrote this anonymous letter?
- Has this document been altered after it was signed?
- Is the handwriting on this cheque consistent with that of the account holder?

For decades, these questions have been answered by qualified forensic document examiners through careful visual inspection and comparison of samples. Today, artificial intelligence is offering a new set of tools that can support, accelerate, and in some cases improve this process — without replacing the expert's judgement.

---

## The Limits of Traditional Manual Analysis

The work of a forensic document examiner is painstaking, requiring years of specialised training and a sharp eye for detail. It is also, by its nature, subject to certain structural limitations:

**Time and scalability.** Manually comparing dozens of handwriting samples — or searching through hundreds of pages of documents for signatures — is a slow process. In large investigations involving extensive documentary evidence, manual analysis can become a bottleneck.

**Subjectivity.** While experienced examiners reach consistent conclusions in clear-cut cases, borderline cases can yield different opinions from different experts. This is not a failing of the individual examiner, but an inherent feature of any process that relies on visual judgement.

**Reproducibility.** A traditional expert opinion describes what an examiner saw and concluded, but it can be difficult to quantify exactly how a particular feature was measured or weighted. This makes it hard for a third party to independently verify the methodology.

**Scale of modern archives.** Digital transformation means that legal disputes increasingly involve large collections of scanned documents. Screening thousands of pages manually is simply not feasible within realistic timeframes.

---

## Why AI Is a Natural Complement

Artificial intelligence — and in particular the branch known as *computer vision* — works by teaching machines to recognise patterns in images. When applied to handwriting, this capability aligns well with what forensic analysis requires:

**Systematic processing at scale.** An AI system can analyse thousands of document images in the time it would take a human expert to review a handful, applying the same criteria consistently across every sample.

**Objective, reproducible measurements.** Rather than describing a characteristic in qualitative terms ("the slant appears moderate"), an AI system produces a precise numerical value — say, an average slant angle of 12 degrees — that can be independently verified and compared across cases.

**Detection of subtle patterns.** Machine learning models can identify statistical regularities in handwriting that may not be immediately visible to the naked eye, such as micro-variations in stroke pressure or subtle inconsistencies in letter spacing.

**Transparent audit trail.** Every step of an AI-assisted analysis can be logged and documented, supporting the chain of custody requirements that are essential in legal proceedings.

None of this eliminates the need for human expertise. What AI provides is an additional layer of objective analysis that the qualified examiner can use to inform, corroborate, or challenge their conclusions.

---

## At a Glance: AI vs Traditional Analysis

| Aspect | Traditional Method | With AI Support |
|--------|--------------------|-----------------|
| Analysis speed | One examiner, one sample at a time | Thousands of samples processed in parallel |
| Type of measurement | Descriptive ("the signature appears similar") | Numerical and verifiable (e.g. "similarity: 87%") |
| Scalability | Limited by available human time | Scales across archives of thousands of documents |
| Reproducibility | Depends on the individual examiner | Identical on every repetition of the analysis |
| Irreplaceable strength | Context, experience, final judgement | Speed, consistency, quantification |

### In Plain Language

**On speed.** Think of the difference between a detective working alone and one with a full office of assistants: the experienced detective makes the final call, but the assistants prepare the case files in seconds rather than days.

**On objectivity.** Instead of writing in a report "this signature looks different from the original", the system produces a concrete figure: "similarity score: 34 out of 100". A number can be challenged and verified in court; a visual impression, however authoritative, is harder to defend under cross-examination.

**On scalability.** Finding a specific signature among 10,000 scanned document pages would take weeks of manual work. An AI system does it in minutes — just as a search engine finds a word across billions of web pages.

---

## Six Areas of Application

### 1. Automatic Transcription of Handwritten Text

When investigators encounter a handwritten letter or document, the first task is often simply to read it. Handwriting can be difficult to decipher, especially in historical documents or cases involving unusual scripts or deteriorated paper.

AI-powered *handwriting recognition* (also known as HTR — Handwritten Text Recognition) converts images of handwritten text directly into machine-readable text, much like a very sophisticated human typist. This has immediate practical value:

- Anonymous letters can be transcribed instantly and searched for keywords
- Historical court documents can be digitised and made searchable
- Transcripts can be produced as a preliminary step before deeper forensic analysis

What AI transcription does **not** do: it does not interpret the meaning or authorship of the text. It simply converts image to words.

---

### 2. Signature Authenticity Verification

Forged signatures are at the heart of many types of legal disputes — from fraudulent cheques to falsified contracts and contested wills. Determining whether a signature is genuine requires comparing it against known authentic samples from the same individual.

AI approaches this problem by learning what makes two signatures look "the same" or "different" — not just in terms of overall shape, but in terms of the fine stylistic details that are consistent in genuine signatures and typically absent in forgeries.

The system produces a **similarity score** — a number indicating how closely the questioned signature matches the reference samples — along with a confidence level. A high similarity score supports a conclusion of authenticity; a low score raises flags for closer examination.

What AI does **not** do: it does not deliver a final legal verdict. It provides a quantitative input that the examiner weighs alongside other evidence.

---

### 3. Automatic Detection of Signatures in Documents

Before a signature can be verified, it must be located. In a multi-page contract, a property deed, or a collection of financial documents, signatures may appear anywhere on the page — sometimes in margins, sometimes partially obscured.

AI-powered *object detection* can automatically scan a document image and identify all regions that contain signatures, drawing a frame around each one. This serves as the first step in an automated analysis pipeline:

1. Detect all signatures in a document
2. Extract each signature as a separate image
3. Send extracted signatures to the verification system

This capability is particularly valuable when processing large document archives, where manual page-by-page inspection is impractical.

---

### 4. Writer Identification

Just as no two people have identical fingerprints, no two people have identical handwriting. The combination of letter forms, spacing habits, pressure patterns, and stylistic quirks that a writer develops over years of practice is unique to that individual.

Writer identification systems analyse these characteristics in an unknown handwriting sample and compare them against a database of reference samples from known individuals. The output is a ranked list of candidate authors, each accompanied by a probability score indicating how closely their style matches the questioned sample.

Applications include:

- Identifying the author of anonymous threatening letters
- Attributing authorship in cases where a document is disputed between parties
- Supporting historical research into the provenance of manuscripts

What AI does **not** do: even a high probability score is not proof of authorship. It is a statistical indication that warrants further expert examination.

---

### 5. Graphological Feature Analysis

Beyond identifying who wrote a document, forensic analysis often requires characterising *how* it was written. Measurable features of handwriting can provide evidence about the writer's physical state, consistency with other known samples, or whether different parts of a document were written under different conditions.

AI tools can automatically extract and quantify the following characteristics:

- **Letter slant:** the angle at which letters lean relative to the vertical. Consistent slant is a marker of a writer's habitual style; inconsistencies may indicate disguise or a different writer.
- **Stroke pressure:** inferred from the darkness and width of ink strokes. Heavier pressure may indicate a different pen, different emotional state, or a different writer.
- **Letter size:** the height and width of individual characters, which tend to be consistent within a writer.
- **Word and letter spacing:** the rhythm of white space between words and letters, which reflects writing speed and habit.
- **Baseline regularity:** how closely letters follow the horizontal baseline, which can vary with fatigue or stress.

These measurements are presented as a visual dashboard, making it straightforward to compare two samples side by side and highlight where they differ.

---

### 6. Named Entity Recognition in Handwritten Documents

A transcribed document often contains names, places, and organisations that are directly relevant to an investigation. Manually reading through long transcripts to identify these entities is time-consuming and error-prone.

*Named Entity Recognition* (NER) is an AI technique that automatically identifies and classifies named entities within a text — persons (PER), organisations (ORG), locations (LOC), and other relevant categories — using models trained on large multilingual corpora.

In a forensic workflow, NER is applied as the second step after HTR transcription:

1. **HTR:** handwritten image → transcribed text
2. **NER:** transcribed text → structured list of entities with confidence scores

This enables investigators to:

- Instantly identify all persons, places, and organisations mentioned in an anonymous letter or contested document
- Build a relationship graph linking entities across a large document corpus
- Screen thousands of pages for the presence of specific names without reading every word

Modern multilingual NER models support Italian, English, German, Spanish, and other languages, making them well-suited for international forensic investigations.

What AI does **not** do: NER identifies entities by their linguistic form, not by their legal significance. The examiner must interpret which entities are relevant and how they relate to the case.

---

## AI as a Support Tool, Not a Replacement

It is important to be clear about what AI analysis is and what it is not.

AI systems produce **probabilistic outputs** — scores, rankings, and confidence intervals — not definitive verdicts. They are trained on historical data and can reflect the biases or limitations of that data. They perform best on clear, high-resolution images and may struggle with degraded or unusual documents.

Most importantly, the interpretation of AI outputs requires the expertise of a qualified forensic document examiner. The examiner brings contextual knowledge, professional judgement, and legal accountability that no AI system can replicate.

The appropriate model is **human-AI collaboration**: AI handles the labour-intensive, quantitative aspects of the analysis, while the expert focuses on interpretation, contextualisation, and the formulation of a professional opinion that meets the evidentiary standards of the legal system.

---

## Ethical and Legal Considerations

The use of AI in forensic contexts raises important questions that must be addressed transparently:

**Bias in training data.** AI models learn from the data they are trained on. If that data over-represents certain populations, languages, or writing styles, the model may perform less reliably on under-represented groups.

**Explainability.** Courts require that expert conclusions be justified and open to challenge. AI systems must therefore be used in ways that can be explained in clear, non-technical terms to judges, juries, and opposing counsel.

**Chain of custody.** Digital evidence must be handled with the same rigour as physical evidence. This means documenting every step of AI-assisted analysis and preserving original files in unaltered form.

**Applicable standards.** Reference frameworks such as those published by OSAC (Organization of Scientific Area Committees for Forensic Science) and SWGDOC (Scientific Working Group for Questioned Documents) provide guidance on best practices for forensic document examination, including the integration of computational tools.

---

## Conclusion

The application of artificial intelligence to forensic graphology represents a significant step forward — not because it replaces the expert examiner, but because it makes their work more rigorous, more efficient, and more reproducible.

When a signature needs to be verified, an anonymous letter attributed, or a large document archive screened for relevant evidence, AI tools can compress what would take days into minutes, and replace subjective visual impressions with precise numerical measurements.

The result is not a machine that delivers verdicts, but a more powerful instrument in the hands of the qualified professional — one that supports the pursuit of truth in the service of justice.

---

*This document accompanies the GraphoLab demo laboratories. For a technical description of each lab, see [NOTEBOOKS_GUIDE.md](NOTEBOOKS_GUIDE.md).*
