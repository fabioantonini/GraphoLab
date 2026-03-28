# GraphoLab: quando l'intelligenza artificiale incontra la grafologia forense

<!--
COME PUBBLICARE SU LINKEDIN:
1. Apri https://markdowntolinkedin.com
2. Incolla il contenuto di questo file (dal titolo in poi)
3. Copia il testo formattato risultante
4. Su LinkedIn → "Scrivi un articolo" → incolla il testo
5. Carica docs/linkedin_cover.png come immagine di copertina (1920×1080)
6. Usa il titolo H1 come titolo dell'articolo LinkedIn
-->

---

Immaginate la scena: uno studio legale, un fascicolo di testamento, e un perito grafologo con la lente d'ingrandimento in mano. La domanda è semplice, ma la risposta può valere milioni di euro — o cambiare il destino di un processo penale: **questa firma è autentica?**

Per decenni, rispondere a questa domanda ha richiesto anni di esperienza, ore di lavoro manuale e, inevitabilmente, un certo grado di soggettività. Oggi, l'intelligenza artificiale offre strumenti nuovi per supportare, accelerare e rendere più rigoroso questo processo.

Ho costruito **GraphoLab** — una raccolta di otto laboratori dimostrativi open source che mostrano come machine learning e computer vision possono essere applicati alla grafologia forense.

---

## Il problema: i limiti dell'analisi tradizionale

La grafologia forense è una disciplina seria e consolidata. Ma come ogni processo manuale, soffre di alcune limitazioni strutturali:

- **Soggettività**: due periti esperti possono giungere a conclusioni diverse sullo stesso documento
- **Scalabilità**: trovare una firma specifica in diecimila pagine scansionate richiede settimane
- **Riproducibilità**: le osservazioni qualitative ("la pressione del tratto sembra diversa") sono difficili da verificare in modo indipendente
- **Velocità**: in procedimenti giudiziari complessi, i tempi dell'analisi manuale possono diventare un collo di bottiglia

L'AI non elimina questi problemi — ma li riduce significativamente, portando misurazioni oggettive accanto all'esperienza umana.

---

## La soluzione: GraphoLab

GraphoLab è un progetto open source che integra sette tecnologie AI in una piattaforma dimostrativa unificata, accessibile via browser grazie a un'app **Gradio**. Ogni laboratorio affronta un compito specifico della grafologia forense.

### Gli otto laboratori

| Lab | Funzionalità | Tecnologia AI |
|-----|-------------|---------------|
| 01 | Introduzione concettuale | — |
| 02 | Riconoscimento testo manoscritto (HTR) | TrOCR / EasyOCR |
| 03 | Verifica autenticità firma | SigNet (Siamese Network) |
| 04 | Rilevamento firma nei documenti | YOLOv8 fine-tuned |
| 05 | Identificazione dello scrittore | HOG + LBP + SVM |
| 06 | Analisi caratteristiche grafologiche | OpenCV + signal processing |
| 07 | Riconoscimento entità nominate (NER) | BERT-NER multilingue |
| 08 | OCR avanzato su corsivo italiano | dots.ocr (VLM 1.7B) |

---

## Approfondimento 1: verificare una firma senza conoscere il firmatario

Una delle domande più frequenti che ricevo: *"SigNet funziona solo con firme già presenti nel database di training?"*

La risposta è no — e la ragione è architetturale.

SigNet usa il **metric learning** (rete siamese con contrastive loss), non un classificatore tradizionale. Un classificatore impara "chi è la persona X" e non può generalizzare a identità mai viste. Un modello metrico, invece, impara a rispondere a una domanda diversa: *"queste due firme provengono dalla stessa mano?"*

Questa domanda è **indipendente dall'identità** del firmatario. SigNet può quindi essere applicata a qualsiasi coppia di firme — anche di persone mai viste durante il training — producendo uno score di similarità coseno che il perito può utilizzare come evidenza quantitativa.

**Le limitazioni esistono e vanno comunicate chiaramente:**
- Il training set (GPDS) contiene principalmente firme brasiliane/portoghesi: stili molto distanti dalla distribuzione di training potrebbero essere meno accurati
- La soglia decisionale (0.35) è calibrata su CEDAR e potrebbe richiedere aggiustamenti per nuovi contesti
- Il sistema è uno strumento di screening, non una prova autonoma: il giudizio finale spetta sempre al perito qualificato

---

## Approfondimento 2: trascrivere il corsivo italiano è un problema aperto

Non tutti gli OCR sono uguali — e nel contesto forense italiano, la differenza può essere decisiva.

**EasyOCR** (usato nell'app interattiva) usa un'architettura CNN + BiLSTM + CTC: veloce (1-3 secondi per immagine su CPU), ma con contesto linguistico limitato. Funziona bene su testo stampato e corsivo regolare.

**TrOCR** (Lab 02) è un Transformer puro: encoder visivo BEiT + decoder RoBERTa. Il contesto linguistico globale (self-attention) lo rende più accurato su corsivo complesso, ma richiede 10-20 secondi per immagine su CPU.

**dots.ocr** (Lab 08) è un Vision-Language Model da 1,7 miliardi di parametri. Il componente LLM corregge le ambiguità visive usando il contesto semantico della frase — risultato: la migliore accuratezza disponibile pubblicamente su corsivo italiano, a fronte di ~7 GB di RAM e 2-5 minuti per immagine su CPU.

La scelta dello strumento giusto dipende dal contesto: per una demo interattiva, EasyOCR. Per la trascrizione forense di un testamento olografo, dots.ocr.

---

## La demo interattiva: sei tab, tutto in un browser

L'app Gradio di GraphoLab aggrega tutte le funzionalità in un'interfaccia accessibile senza installazione (con Docker):

- **OCR Manoscritto** — carica un'immagine, ottieni il testo trascritto
- **Verifica Firma** — carica due firme, ottieni il verdetto autentica/falsa con score
- **Rilevamento Firma** — carica un documento multi-pagina, estrai automaticamente tutte le firme
- **Riconoscimento Entità** — identifica persone, luoghi, organizzazioni nel testo
- **Identificazione Scrittore** — attribuisci la paternità di un campione anonimo
- **Analisi Grafologica** — misura inclinazione, spaziatura, pressione del tratto
- **Pipeline Forense** — referto completo in un unico passaggio
- **Datazione Documenti** — carica più documenti, ottieni l'ordine cronologico per data estratta

---

## Etica e limiti: l'AI non sostituisce il perito

Questo punto merita di essere detto chiaramente.

GraphoLab è uno strumento di supporto, non un oracolo. I modelli AI producono scores e misurazioni — non verdetti. Il modello appropriato è la **collaborazione uomo-AI**: l'AI gestisce gli aspetti quantitativi e ad alta intensità di lavoro dell'analisi, mentre l'esperto si concentra sull'interpretazione, la contestualizzazione e la responsabilità legale.

In un'aula di tribunale, "il modello dice X" non è una prova. "Il perito ha utilizzato questo strumento per corroborare la propria analisi, ecco come" è un'altra storia.

La trasparenza sui limiti dei modelli — dataset di training, soglie di decisione, condizioni di validità — è parte integrante di un uso responsabile di questi strumenti in ambito forense.

---

## Provalo tu stesso

GraphoLab è completamente open source, con licenza Apache 2.0.

**Repository GitHub:** https://github.com/fabioantonini/GraphoLab

Trovi:
- Otto Jupyter notebook eseguibili (Python 3.11/3.12 + PyTorch)
- L'app Gradio avviabile in locale o via Docker
- Documentazione completa in italiano e inglese
- Script per generare dati di test sintetici

```bash
git clone https://github.com/fabioantonini/GraphoLab.git
cd GraphoLab
docker compose up gradio
# Apri http://localhost:7860
```

---

Sono curioso di sapere cosa ne pensate — in particolare chi lavora nel settore forense o legale. L'AI in questo ambito è ancora largamente inesplorata in Italia, e credo ci siano opportunità significative per chi vuole portare rigore quantitativo nel lavoro peritale.

**Grafologi, avvocati, notai, informatici forensi: cosa vi aspettate da strumenti come questo? Cosa manca ancora?**

---

*Fabio Antonini — AI Engineer & Researcher*
*GitHub: https://github.com/fabioantonini/GraphoLab*
