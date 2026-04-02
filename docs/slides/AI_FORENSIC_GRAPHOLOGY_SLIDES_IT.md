---
marp: true
theme: gaia
class: invert
paginate: true
footer: "GraphoLab — AI e Grafologia Forense"
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

# Intelligenza Artificiale e<br>Grafologia Forense

### Come il machine learning sta trasformando<br>l'analisi della scrittura e delle firme

<br>

**GraphoLab** — Laboratorio Dimostrativo

---

## Agenda

1. Cos'è la grafologia forense?
2. I limiti dell'analisi manuale tradizionale
3. Perché l'AI è un complemento naturale
4. Cinque aree di applicazione
5. L'AI come strumento di supporto — non sostituto
6. Considerazioni etiche e legali
7. I laboratori GraphoLab

---

<!-- _class: divider -->

# Parte 1
## Il Dominio

---

## Cos'è la Grafologia Forense?

> *L'esame scientifico della scrittura a mano e delle firme per rispondere a domande rilevanti in ambito giudiziario.*

Le domande tipiche in sede legale:

- Questa **firma è autentica**, o è stata **contraffatta**?
- **Chi ha scritto** questa lettera anonima?
- Questo **documento è stato alterato** dopo la firma?
- La scrittura è **coerente** con l'autore presunto?

---

## Il Flusso di Lavoro Tradizionale

Il perito calligrafo segue questi passaggi:

| Fase | Attività |
|------|----------|
| **Acquisizione** | Scansione del documento a 600–1200 DPI |
| **Raccolta campioni** | Campioni noti dai potenziali autori |
| **Confronto visivo** | Inclinazione, pressione, spaziatura, forme delle lettere |
| **Parere peritale** | Redazione della relazione per il tribunale |

---

<!-- _class: divider -->

# Parte 2
## I Limiti dell'Analisi Manuale

---

## Dove l'Analisi Manuale Incontra i Suoi Limiti

**Tempo e scalabilità**
Confrontare decine di campioni o cercare firme in centinaia di pagine è lento. Nelle indagini complesse diventa un collo di bottiglia.

**Soggettività**
I casi borderline possono dar luogo a opinioni divergenti tra periti diversi — non per incapacità, ma per i limiti intrinseci del giudizio visivo.

**Riproducibilità**
Le relazioni tradizionali sono qualitative. Terze parti non possono facilmente verificare *come* una caratteristica è stata misurata o *quanto peso* le è stato attribuito.

**Scala degli archivi moderni**
Le controversie legali coinvolgono sempre più spesso migliaia di documenti scansionati. L'esame manuale pagina per pagina non è praticabile.

---

<!-- _class: divider -->

# Parte 3
## Perché l'AI è un Complemento Naturale

---

## L'AI Aggiunge Ciò che Manca all'Analisi Manuale

| Criticità | Contributo dell'AI |
|-----------|-------------------|
| Velocità | Migliaia di immagini elaborate in minuti |
| Obiettività | Misurazioni numeriche precise, non impressioni |
| Riproducibilità | Stesso algoritmo, stesso risultato — sempre |
| Scala | Grandi archivi esaminati automaticamente |
| Trasparenza | Ogni passo registrato → audit trail completo |

---

## Come l'AI "Legge" la Scrittura

L'AI usa la **computer vision** — insegna alle macchine a riconoscere schemi nelle immagini.

Applicata alla scrittura, significa:

- Rilevare **regolarità sottili** invisibili a occhio nudo
- Misurare le caratteristiche **con precisione e coerenza**
- Confrontare i campioni **statisticamente**, non solo visivamente
- Produrre output **quantificabili e spiegabili**

> La scrittura a mano è ricca di schemi. L'AI è molto brava con gli schemi.

---

<!-- _class: divider -->

# Parte 4
## Cinque Aree di Applicazione

---

## 1. Trascrizione Automatica di Testi Manoscritti

**Cosa fa**
Converte direttamente le immagini di testo manoscritto in testo leggibile dalla macchina — come un dattilografo molto sofisticato.

**Perché è utile forensicamente**
- Lettere anonime → trascrizione immediata e ricerca per parole chiave
- Documenti giudiziari storici → digitalizzati e ricercabili
- Documenti lunghi → elaborati in secondi, non in ore

**Cosa NON fa**
Non interpreta il significato né attribuisce l'autoria. Converte l'immagine in parole.

> *Lab 02 in GraphoLab*

---

## 2. Verifica dell'Autenticità della Firma

**Cosa fa**
Confronta una firma in esame con campioni autentici di riferimento e restituisce un **punteggio di similarità** con un livello di confidenza.

**Perché è utile forensicamente**
- Assegni bancari, contratti, testamenti, atti legali
- Rileva firme tracciate o riprodotte digitalmente
- Quantifica quanto una firma contestata si discosta dal riferimento

**Cosa NON fa**
Non emette un verdetto legale. Fornisce un dato quantitativo per il perito.

> *Lab 03 in GraphoLab*

---

## 3. Rilevamento Firma nei Documenti

**Cosa fa**
Scansiona automaticamente un documento e **localizza tutte le firme**, tracciando un riquadro attorno a ciascuna.

**Perché è utile forensicamente**
- Contratti multi-pagina → firme trovate istantaneamente
- Grandi archivi → screening automatico in blocco
- Alimenta la pipeline: **rileva → estrai → verifica**

**Cosa NON fa**
Non distingue firme autentiche da contraffatte — quello è il passo successivo.

> *Lab 04 in GraphoLab*

---

## 4. Identificazione dello Scrittore

**Cosa fa**
Analizza un campione manoscritto anonimo e restituisce una **lista ordinata di autori candidati** con punteggi di probabilità, confrontando un database di campioni di riferimento.

**Perché è utile forensicamente**
- Lettere minatorie anonime → candidati autori identificati
- Paternità di documenti contestata → evidenza statistica di attribuzione
- Manoscritti storici → ricerca sulla provenienza

**Cosa NON fa**
Un punteggio di probabilità elevato non è prova dell'autoria. È un'indicazione statistica per ulteriore esame peritale.

> *Lab 05 in GraphoLab*

---

## 5. Analisi delle Caratteristiche Grafologiche

**Cosa fa**
Estrae e quantifica automaticamente le caratteristiche grafologiche di un'immagine di testo manoscritto.

| Caratteristica | Cosa rivela |
|---------------|-------------|
| **Inclinazione lettere** | Stile abituale; tentativi di camuffamento |
| **Pressione del tratto** | Tipo di penna, stato fisico |
| **Dimensione lettere** | Coerenza tra campioni diversi |
| **Spaziatura parole** | Velocità e abitudini di scrittura |
| **Regolarità del rigo** | Stanchezza, indicatori di stress |

> *Lab 06 in GraphoLab*

---

<!-- _class: divider -->

# Parte 5
## L'AI come Strumento di Supporto

---

## Collaborazione Uomo + AI

L'AI non sostituisce il perito forense. È uno **strumento più potente** nelle sue mani.

```
L'AI gestisce:                  Il perito gestisce:
──────────────────              ────────────────────────────
• Scala di elaborazione         • Interpretazione contestuale
• Misurazioni numeriche         • Giudizio professionale
• Coerenza e ripetibilità       • Responsabilità legale
• Audit trail                   • Testimonianza in tribunale
```

**Il modello:** l'AI comprime il lavoro quantitativo così il perito si concentra su ciò che solo un umano può fare.

---

## Cosa Sono — e Cosa Non Sono — gli Output AI

| L'AI produce | L'AI NON produce |
|-------------|-----------------|
| **Punteggi** di similarità | **Verdetti** legali |
| **Classifiche** di probabilità | Prova dell'autoria |
| **Misurazioni** numeriche | Opinione peritale conclusiva |
| **Indicatori** statistici | Giudizio contestuale |

> Tutti gli output AI richiedono la validazione di un perito calligrafo qualificato prima dell'uso in giudizio.

---

<!-- _class: divider -->

# Parte 6
## Considerazioni Etiche e Legali

---

## Aspetti Fondamentali

**Bias nei dati di addestramento**
I modelli addestrati su popolazioni limitate possono funzionare in modo meno affidabile su stili di scrittura o lingue sotto-rappresentati.

**Spiegabilità**
I tribunali richiedono conclusioni giustificabili. I sistemi AI devono poter essere spiegati chiaramente a giudici, giurie e controparti.

**Catena di custodia**
Le prove digitali devono essere gestite con rigore: ogni passo documentato, i file originali preservati inalterati.

**Standard di riferimento**
OSAC e SWGDOC pubblicano linee guida sulle migliori pratiche per l'esame forense dei documenti, inclusa l'integrazione di strumenti computazionali.

---

<!-- _class: divider -->

# Parte 7
## I Laboratori GraphoLab

---

## GraphoLab — Panoramica dei Lab

| Lab | Titolo | Tecnologia |
|-----|--------|-----------|
| 01 | Introduzione | Panoramica concettuale |
| 02 | OCR Manoscritto | TrOCR (Transformer) |
| 03 | Verifica Firma | Siamese Neural Network |
| 04 | Rilevamento Firma | Conditional DETR Object Detection |
| 05 | Identificazione Scrittore | Estrazione feature + SVM |
| 06 | Analisi Grafologica | OpenCV + Elaborazione immagini |
| — | Demo Gradio | Tutte le funzionalità, via browser |

Ogni lab include codice funzionante, dati sintetici di fallback e una sezione sui casi d'uso forensi.

---

## Come Avviare i Laboratori

**In locale (Python + JupyterLab)**
```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

**Docker — JupyterLab**
```bash
docker compose up jupyter
# http://localhost:8888  (token: grapholab)
```

**Docker — Demo Interattiva**
```bash
docker compose up gradio
# http://localhost:7860
```

---

<!-- _class: lead -->

# Conclusione

L'AI non sostituisce il perito forense.

Rende il suo lavoro **più rapido**, **più oggettivo** e **più riproducibile**.

<br>

L'obiettivo non è una macchina che emette verdetti —
ma uno strumento più potente nelle mani
del professionista qualificato.

<br>

**GraphoLab** · [github.com/fabioantonini/GraphoLab](https://github.com/fabioantonini/GraphoLab)
