# Intelligenza Artificiale e Grafologia Forense

## Perché l'AI Sta Trasformando l'Analisi della Scrittura e delle Firme

---

## Introduzione

Ogni volta che una persona scrive a mano o appone una firma su un documento, lascia una traccia tanto personale e distintiva quanto un'impronta digitale. Il modo in cui le lettere vengono tracciate, l'angolazione della penna, la pressione esercitata sul foglio, il ritmo dei tratti: tutte queste caratteristiche formano un profilo stilistico unico che si mantiene costante nei diversi campioni e nel tempo.

La **grafologia forense** — più formalmente denominata *esame dei documenti contestati* (*questioned document examination*) — è la disciplina scientifica che studia queste caratteristiche per rispondere a domande rilevanti in ambito giudiziario:

- Questa firma è autentica, o è stata contraffatta?
- Chi ha scritto questa lettera anonima?
- Questo documento è stato alterato dopo la firma?
- La scrittura su questo assegno è coerente con quella del titolare del conto?

Per decenni, queste domande sono state affrontate da periti calligrafi qualificati attraverso l'ispezione visiva e il confronto dei campioni. Oggi l'intelligenza artificiale offre un nuovo insieme di strumenti in grado di supportare, accelerare e in alcuni casi migliorare questo processo — senza sostituire il giudizio dell'esperto.

---

## I Limiti dell'Analisi Manuale Tradizionale

Il lavoro del perito calligrafo è minuzioso, richiede anni di formazione specializzata e un occhio particolarmente allenato al dettaglio. È anche, per sua natura, soggetto ad alcune limitazioni strutturali:

**Tempo e scalabilità.** Confrontare manualmente decine di campioni di scrittura — o cercare firme in centinaia di pagine di documenti — è un processo lento. Nelle indagini più complesse, che coinvolgono un'ampia documentazione, l'analisi manuale può diventare un collo di bottiglia.

**Soggettività.** Sebbene i periti esperti raggiungano conclusioni coerenti nei casi più chiari, le situazioni borderline possono dar luogo a opinioni divergenti tra diversi esperti. Non si tratta di una mancanza individuale, ma di una caratteristica intrinseca di qualsiasi processo basato sul giudizio visivo.

**Riproducibilità.** Una perizia tradizionale descrive ciò che l'esaminatore ha osservato e concluso, ma è difficile quantificare con precisione come un determinato elemento è stato misurato o ponderato. Questo rende arduo per terze parti verificare in modo indipendente la metodologia adottata.

**Scala degli archivi moderni.** La trasformazione digitale ha fatto sì che le controversie legali coinvolgano sempre più spesso grandi raccolte di documenti scansionati. Esaminare manualmente migliaia di pagine non è realizzabile entro tempi ragionevoli.

---

## Perché l'AI è un Complemento Naturale

L'intelligenza artificiale — e in particolare il ramo noto come *computer vision* — opera insegnando alle macchine a riconoscere schemi nelle immagini. Applicata alla scrittura, questa capacità si allinea naturalmente con le esigenze dell'analisi forense:

**Elaborazione sistematica su larga scala.** Un sistema AI può analizzare migliaia di immagini di documenti nel tempo in cui un perito umano ne esaminerebbe una manciata, applicando gli stessi criteri in modo uniforme a ogni campione.

**Misurazioni oggettive e riproducibili.** Invece di descrivere una caratteristica in termini qualitativi ("l'inclinazione appare moderata"), un sistema AI produce un valore numerico preciso — per esempio un'inclinazione media di 12 gradi — che può essere verificato indipendentemente e confrontato tra casi diversi.

**Rilevamento di pattern sottili.** I modelli di machine learning possono identificare regolarità statistiche nella scrittura che potrebbero non essere immediatamente visibili a occhio nudo, come micro-variazioni nella pressione del tratto o sottili incongruenze nella spaziatura delle lettere.

**Audit trail trasparente.** Ogni fase di un'analisi assistita da AI può essere registrata e documentata, supportando i requisiti di catena di custodia fondamentali nei procedimenti legali.

Nulla di tutto questo elimina la necessità dell'esperienza umana. Ciò che l'AI fornisce è un ulteriore livello di analisi oggettiva che il perito qualificato può utilizzare per informare, corroborare o contestare le proprie conclusioni.

---

## Sei Aree di Applicazione

### 1. Trascrizione Automatica di Testi Manoscritti

Quando gli investigatori si trovano di fronte a una lettera o un documento manoscritto, il primo compito è spesso semplicemente leggerlo. La scrittura a mano può essere difficile da decifrare, specialmente in documenti storici o in presenza di grafie insolite o supporti deteriorati.

Il *riconoscimento automatico della scrittura a mano* (HTR — Handwritten Text Recognition) converte direttamente le immagini di testo manoscritto in testo leggibile dalla macchina, come farebbe un dattilografo molto sofisticato. Questo ha un valore pratico immediato:

- Le lettere anonime possono essere trascritte istantaneamente e ricercate per parole chiave
- I documenti giudiziari storici possono essere digitalizzati e resi ricercabili
- Le trascrizioni possono essere prodotte come fase preliminare a un'analisi forense più approfondita

Cosa la trascrizione AI **non** fa: non interpreta il significato né l'autoria del testo. Si limita a convertire l'immagine in parole.

---

### 2. Verifica dell'Autenticità della Firma

Le firme contraffatte sono al centro di molti tipi di controversie legali — dagli assegni fraudolenti ai contratti falsificati, fino ai testamenti contestati. Stabilire se una firma è autentica richiede di confrontarla con campioni autentici noti dello stesso individuo.

I sistemi AI affrontano questo problema imparando cosa rende due firme "uguali" o "diverse" — non solo in termini di forma complessiva, ma nei dettagli stilistici sottili che sono coerenti nelle firme genuine e tipicamente assenti nelle contraffazioni.

Il sistema produce un **punteggio di similarità** — un numero che indica quanto la firma in esame corrisponda ai campioni di riferimento — insieme a un livello di confidenza. Un punteggio di similarità elevato supporta una conclusione di autenticità; un punteggio basso segnala la necessità di un esame più approfondito.

Cosa l'AI **non** fa: non emette un verdetto legale definitivo. Fornisce un dato quantitativo che il perito valuta insieme ad altri elementi probatori.

---

### 3. Rilevamento Automatico di Firme nei Documenti

Prima che una firma possa essere verificata, deve essere localizzata. In un contratto multi-pagina, un atto notarile o una raccolta di documenti finanziari, le firme possono comparire in qualsiasi punto della pagina — a volte nei margini, a volte parzialmente coperte.

I sistemi AI di *rilevamento degli oggetti* possono scansionare automaticamente un'immagine di documento e identificare tutte le aree che contengono firme, tracciando un riquadro attorno a ciascuna. Questo costituisce il primo passo di una pipeline di analisi automatizzata:

1. Rilevare tutte le firme nel documento
2. Estrarre ciascuna firma come immagine separata
3. Inviare le firme estratte al sistema di verifica

Questa funzionalità è particolarmente preziosa nell'elaborazione di grandi archivi documentali, dove l'ispezione manuale pagina per pagina è impraticabile.

---

### 4. Identificazione dello Scrittore

Così come non esistono due persone con impronte digitali identiche, non esistono due persone con una grafia identica. La combinazione di forme delle lettere, abitudini di spaziatura, schemi di pressione e idiosincrasie stilistiche che uno scrittore sviluppa nel corso di anni di pratica è unica per quell'individuo.

I sistemi di identificazione dello scrittore analizzano queste caratteristiche in un campione di scrittura sconosciuto e le confrontano con un database di campioni di riferimento provenienti da individui noti. Il risultato è una lista ordinata di autori candidati, ciascuno accompagnato da un punteggio di probabilità che indica quanto il loro stile corrisponda al campione in esame.

Le applicazioni includono:

- Identificazione dell'autore di lettere minatorie anonime
- Attribuzione dell'autoria in casi in cui un documento è contestato tra le parti
- Supporto alla ricerca storica sulla provenienza di manoscritti

Cosa l'AI **non** fa: anche un punteggio di probabilità elevato non costituisce prova dell'autoria. È un'indicazione statistica che giustifica un ulteriore esame peritale.

---

### 5. Analisi delle Caratteristiche Grafologiche

Oltre a identificare chi ha scritto un documento, l'analisi forense richiede spesso di caratterizzare *come* è stato scritto. Le caratteristiche misurabili della scrittura possono fornire elementi probatori riguardo allo stato fisico dello scrittore, alla coerenza con altri campioni noti, o all'eventualità che diverse parti di un documento siano state scritte in condizioni diverse.

Gli strumenti AI possono estrarre e quantificare automaticamente le seguenti caratteristiche:

- **Inclinazione delle lettere:** l'angolo con cui le lettere si inclinano rispetto alla verticale. Un'inclinazione costante è un indicatore dello stile abituale dello scrittore; le incongruenze possono segnalare un tentativo di camuffamento o la presenza di uno scrittore diverso.
- **Pressione del tratto:** dedotta dall'intensità e dallo spessore dei tratti d'inchiostro. Una pressione diversa può indicare una penna diversa, uno stato emotivo differente o un autore diverso.
- **Dimensione delle lettere:** l'altezza e la larghezza dei singoli caratteri, che tendono a essere coerenti all'interno della grafia di uno stesso scrittore.
- **Spaziatura tra parole e lettere:** il ritmo degli spazi bianchi, che riflette la velocità di scrittura e le abitudini individuali.
- **Regolarità del rigo di base:** quanto le lettere seguono la linea di base orizzontale, variabile con la stanchezza o lo stress.

Queste misurazioni vengono presentate come una dashboard visiva, che rende immediato il confronto tra due campioni e l'evidenziazione delle differenze.

---

### 6. Riconoscimento delle Entità Nominate nei Documenti Manoscritti

Un documento trascritto contiene spesso nomi di persone, luoghi e organizzazioni direttamente rilevanti per un'indagine. Leggere manualmente trascrizioni estese per identificare queste entità è lento e soggetto a omissioni.

Il *Riconoscimento delle Entità Nominate* (NER — Named Entity Recognition) è una tecnica AI che identifica e classifica automaticamente le entità nominate all'interno di un testo — persone (PER), organizzazioni (ORG), luoghi (LOC) e altre categorie pertinenti — utilizzando modelli addestrati su grandi corpus multilingue.

In un flusso di lavoro forense, il NER viene applicato come secondo passo dopo la trascrizione HTR:

1. **HTR:** immagine manoscritta → testo trascritto
2. **NER:** testo trascritto → lista strutturata di entità con punteggi di confidenza

Questo consente agli investigatori di:

- Identificare istantaneamente tutte le persone, i luoghi e le organizzazioni menzionati in una lettera anonima o in un documento contestato
- Costruire un grafo delle relazioni che colleghi le entità attraverso un corpus documentale esteso
- Esaminare migliaia di pagine alla ricerca di nomi specifici senza doverle leggere integralmente

I moderni modelli NER multilingue supportano italiano, inglese, tedesco, spagnolo e altre lingue, rendendoli adatti a indagini forensi internazionali.

Cosa l'AI **non** fa: il NER identifica le entità in base alla loro forma linguistica, non alla loro rilevanza giuridica. L'interpretazione di quali entità siano pertinenti e come si colleghino al caso resta compito del perito.

---

## L'AI come Strumento di Supporto, Non un Sostituto

È importante essere chiari su cosa sia e cosa non sia l'analisi AI.

I sistemi AI producono **output probabilistici** — punteggi, classificazioni e intervalli di confidenza — non verdetti definitivi. Sono addestrati su dati storici e possono riflettere i bias o i limiti di quei dati. Funzionano meglio su immagini chiare e ad alta risoluzione, e possono avere difficoltà con documenti deteriorati o insoliti.

Soprattutto, l'interpretazione degli output AI richiede la competenza di un perito calligrafo qualificato. Il perito porta conoscenza contestuale, giudizio professionale e responsabilità legale che nessun sistema AI può replicare.

Il modello appropriato è la **collaborazione uomo-AI**: l'AI gestisce gli aspetti quantitativi e ad alta intensità di lavoro dell'analisi, mentre l'esperto si concentra sull'interpretazione, la contestualizzazione e la formulazione di un'opinione professionale che rispetti gli standard probatori del sistema giudiziario.

---

## Considerazioni Etiche e Legali

L'uso dell'AI in contesti forensi solleva questioni importanti che devono essere affrontate con trasparenza:

**Bias nei dati di addestramento.** I modelli AI imparano dai dati su cui vengono addestrati. Se quei dati sovra-rappresentano determinate popolazioni, lingue o stili di scrittura, il modello potrebbe funzionare in modo meno affidabile su gruppi sotto-rappresentati.

**Spiegabilità.** I tribunali richiedono che le conclusioni peritali siano giustificate e aperte alla contestazione. I sistemi AI devono quindi essere utilizzati in modi che possano essere spiegati in termini chiari e non tecnici a giudici, giurie e controparti.

**Catena di custodia.** Le prove digitali devono essere gestite con la stessa rigore delle prove fisiche. Ciò significa documentare ogni fase dell'analisi assistita da AI e conservare i file originali in forma inalterata.

**Standard di riferimento.** Framework come quelli pubblicati dall'OSAC (Organization of Scientific Area Committees for Forensic Science) e dallo SWGDOC (Scientific Working Group for Questioned Documents) forniscono linee guida sulle migliori pratiche per l'esame forense dei documenti, inclusa l'integrazione di strumenti computazionali.

---

## Conclusione

L'applicazione dell'intelligenza artificiale alla grafologia forense rappresenta un passo avanti significativo — non perché sostituisca il perito esperto, ma perché rende il suo lavoro più rigoroso, più efficiente e più riproducibile.

Quando una firma deve essere verificata, una lettera anonima attribuita o un grande archivio documentale esaminato alla ricerca di prove rilevanti, gli strumenti AI possono comprimere in pochi minuti ciò che richiederebbe giorni, e sostituire impressioni visive soggettive con misurazioni numeriche precise.

Il risultato non è una macchina che emette verdetti, ma uno strumento più potente nelle mani del professionista qualificato — al servizio della ricerca della verità e della giustizia.

---

*Questo documento accompagna i laboratori dimostrativi di GraphoLab. Per una descrizione tecnica di ciascun laboratorio, vedere [NOTEBOOKS_GUIDE_IT.md](NOTEBOOKS_GUIDE_IT.md).*
