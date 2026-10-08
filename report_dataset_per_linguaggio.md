# Dataset per Vulnerability Detection con DST — Analisi per Linguaggio

## Obiettivo e criteri di valutazione

L'obiettivo è stimare in modo accurato le performance dei tool di analisi statica (SAT) per ciascuna CWE presente nei dataset, per poi usarle come valori di confidence nella **Dempster-Shafer Theory (DST)**. La combinazione DST mira a superare le performance di un singolo tool e di un semplice voting 2-out-of-N.

Per ogni sample sono necessari:
- **Codice sorgente** della funzione — input ai SAT
- **CWE** — per raggruppare i campioni e stimare la confidence per-CWE
- **Label binaria ground truth** (0/1) — per calcolare TP, FP, TN, FN

I criteri di valutazione sono:

- **Label quality**: affidabilità della ground truth
- **Tipo di safe**: *puri* (safe per design, non correlati strutturalmente al vulnerabile) vs *fix-paired* (versione fixed del codice vulnerabile) vs *generici commit* (funzioni non modificate dal fix-commit)
- **Origine**: reale (repo OSS), sintetico, o AI-generated
- **Disponibilità CWE**: diretta, tramite join, o assente/non strutturata

---

## Struttura dello studio: 3 branch paralleli

Lo studio viene condotto parallelamente su tre branch distinti, ciascuno corrispondente alla natura del codice sorgente:

- **Branch Real**: codice estratto da repository open source reali, labellato tramite CVE/commit. Produce stime di confidence applicabili al codice reale in produzione.
- **Branch Synth**: codice sintetico costruito artificialmente (NIST, OWASP). Label perfette per costruzione, ma distribuzione lontana dal codice reale. Produce stime utili come baseline controllata e per CWE rare con pochi campioni reali.
- **Branch AI**: codice generato da LLM. Distribuzione distinta sia dal codice reale che da quello sintetico. Produce stime applicabili specificamente alla valutazione di codice AI-generated, use case sempre più rilevante.

I criteri di esclusione restano indipendenti dall'origine del codice e riguardano esclusivamente: **label quality insufficiente**, **assenza di CWE non ricavabile in alcun modo**, e **circolo vizioso nel labeling** (ground truth prodotto dagli stessi SAT che si vuole valutare). L'assenza di campioni safe non è di per sé un criterio di esclusione: i negativi possono essere recuperati da altri dataset. Informazioni parziali (CWE non strutturata, safe assenti) sono accettabili se il dato mancante è filtrabile o recuperabile.

---

## Note metodologiche sui safe

Il tipo di campione safe è critico per la stima della confidence dei SAT. Un safe *fix-paired* è strutturalmente molto simile al vulnerabile corrispondente (differisce spesso di poche righe), il che può rendere il task artificialmente facile per i SAT e gonfiare la stima di confidence. I safe *puri* — costruiti per design per non contenere la vulnerabilità — rappresentano la situazione più realistica e producono stime più affidabili. Questa distinzione vale trasversalmente a tutti e tre i branch.

---
---

# C / C++

## Tabella riepilogativa

| Dataset | Branch | CWE | Vuln | Safe | Tipo safe | Label quality | Giudizio |
|---|---|---|---|---|---|---|---|
| **ICVul** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐⭐ | ✅ primario |
| **ReposVul** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐⭐ | ✅ primario |
| **MegaVul** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐ | ✅ primario |
| **CVEfixes** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐ | ✅ primario |
| **PrimeVul** | 🟢 Real | ✅ parziale | ✅ | ✅ | generici commit | ⭐⭐⭐⭐ | ✅ primario |
| **DiverseVul** | 🟢 Real | ✅ parziale | ✅ | ✅ | generici commit | ⭐⭐⭐ | ⚠️ secondario |
| **CrossVul** | 🟢 Real | ✅ | ✅ | ✅ | **puri** | ⭐⭐ (48%) | ⚠️ solo safe |
| **SVEN** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐ | ⚠️ secondario |
| **BigVul** | 🟢 Real | ✅ | ✅ | ✅ | generici commit | ⭐ (25%) | ❌ |
| **Juliet C/C++ 1.3** | 🔵 Synth | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐⭐⭐ | ✅ primario |
| **CASTLE** | 🔵 Synth | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐⭐ | ✅ primario |
| **LLMSecEval** | 🟡 AI | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐ | ✅ primario |
| **FormAI** | 🟡 AI | ⚠️ parziale | ✅ | ✅ | n/a | ⭐⭐⭐ | ⚠️ secondario |

---

## Analisi critica per dataset

### 🟢 Real

#### ICVul ✅ primario

È il dataset con la migliore tracciabilità della relazione funzione↔vulnerabilità tra tutti i dataset reali disponibili. La sua caratteristica distintiva è l'uso dell'algoritmo **SZZ** per identificare il **Vulnerability Contributing Commit (VCC)** — non il commit che fixa la vulnerabilità, ma quello che l'ha *introdotta*. Questo permette di identificare quale funzione è effettivamente la sede del bug, riducendo drasticamente il problema delle funzioni irrilevanti etichettate come vulnerabili. La tecnica **ESC (Eliminate Suspicious Commit)** aggiunge un ulteriore livello di filtraggio. La CWE è disponibile nel mapping CVE→CWE incluso nel dataset.

Il principale svantaggio è la struttura relazionale multi-tabella che richiede join non banali per ottenere il triplo (funzione, CWE, label). I safe sono fix-paired.

**Quando usarlo**: fonte primaria di positivi reali per le CWE più rappresentate. Prioritario quando la certezza della corrispondenza funzione↔CWE è critica.

---

#### ReposVul ✅ primario

Dataset pubblicato a ICSE 2024 che affronta direttamente il principale problema strutturale dei dataset commit-based: le **tangled patches**. Il framework di costruzione include tre moduli distinti: un **vulnerability untangling module** che usa LLM e static analysis tools congiuntamente per separare le modifiche realmente legate al fix da quelle non correlate; un **multi-granularity dependency extraction module** che cattura le relazioni inter-procedurali tra funzioni; e un **trace-based filtering module** che elimina patch obsolete. Copre 6.134 CVE entries, 236 tipi di CWE, in 1.491 progetti distribuiti su C, C++, Java e Python, con file separati per linguaggio scaricabili individualmente.

Rispetto a ICVul, che usa SZZ per tracciare il VCC, ReposVul attacca il problema da una prospettiva complementare: invece di trovare il commit che ha introdotto la vulnerabilità, filtra le funzioni irrilevanti dal commit che la fixa. I safe sono fix-paired. La validazione manuale degli autori supporta l'alta label quality dichiarata.

**Quando usarlo**: fonte primaria per tutti e tre i linguaggi (C, Java, Python), in particolare come complemento a ICVul/MegaVul per C e come fonte di alta qualità per Java e Python dove le alternative reali sono scarse. La copertura di 236 CWE lo rende tra i più ampi disponibili.

Evoluzione diretta di BigVul con filtraggio migliorato dei commit irrilevanti e arricchimento di metadati (CVSS score, descrizione CVE, rappresentazioni del codice come AST e PDG). La presenza del **CVSS score** è un vantaggio significativo: permette di filtrare ulteriormente i campioni a bassa severità, statisticamente più soggetti a mislabeling. Copre 169 tipi di CWE e oltre 17.000 funzioni vulnerabili, con le CWE di memoria dominanti (CWE-119, CWE-125, CWE-787 rappresentano circa il 60% dei campioni).

Eredita il problema strutturale del commit-based labeling (funzioni non direttamente vulnerabili etichettate come 1), ma in misura minore rispetto a BigVul grazie al filtraggio più aggressivo. I safe sono fix-paired.

**Quando usarlo**: fonte primaria ad ampia copertura di CWE, in combinazione con ICVul. Particolarmente utile per le CWE di memoria, quelle maggiormente rilevabili dai SAT.

---

#### CVEfixes ✅ primario

Dataset commit-based automatico da NVD con coppie vulnerabile/fixed. Tra i dataset commit-based automatici ha la label accuracy considerata più alta, grazie alla selezione rigorosa dei CVE da NVD. Copre C/C++ come sottoinsieme del dataset multi-linguaggio; la CWE è disponibile tramite il link CVE→NVD nella versione zenodo originale.

**Attenzione**: la versione preprocessata su Kaggle non include la CWE. Per uso per-CWE è indispensabile la versione zenodo originale con join NVD. I safe sono fix-paired.

**Quando usarlo**: complemento a ICVul e MegaVul, soprattutto per CWE con pochi campioni negli altri dataset. Struttura semplice che ne facilita l'integrazione.

---

#### PrimeVul ✅ primario

Versione rifinita di DiverseVul con deduplication rigorosa e splitting cronologico per evitare data leakage. Il problema dei near-duplicate sistemici — che in DiverseVul gonfiavano artificialmente le performance — è risolto. I safe sono generici commit, non puri, ma la deduplication riduce la correlazione strutturale con i positivi. È il dataset usato dalla letteratura recente come benchmark "difficile": le performance realistiche su PrimeVul sono sensibilmente più basse che su DiverseVul, il che per il branch Real produce stime di confidence più affidabili.

**Quando usarlo**: preferire a DiverseVul quando la qualità supera la quantità. Dataset primario per il branch Real quando si vuole evitare l'inflazione delle performance.

---

#### DiverseVul ⚠️ secondario

Dataset ampio (18.945 funzioni vulnerabili, 150 CWE), costruito aggregando BigVul, ReVeal, CrossVul, CVEfixes e Devign con deduplication. La sua forza è la **copertura di CWE rare**: con 150 tipi diversi è il dataset con la gamma più ampia per il branch Real.

Il problema principale è la label accuracy al 60% e la struttura delle colonne `target` e `cwe` che richiede attenzione: la CWE è del commit, non della funzione; i safe con CWE non nulla non sono "safe per quella CWE". La costruzione corretta del subset per-CWE richiede: positivi = `target=1` AND `cwe` contiene la CWE di interesse; negativi = tutti i `target=0` indipendentemente dalla CWE.

**Quando usarlo**: integrazione per CWE rare con pochi campioni nei dataset primari. Non come fonte principale per la stima della confidence.

---

#### CrossVul ⚠️ solo safe

Caso peculiare: label accuracy al 48% sui positivi, ma i safe sono **puri per costruzione** — per ogni file vulnerabile (`bad_X`) esiste un file safe (`good_X`) scritto come alternativa corretta, non come patch. Questo li rende strutturalmente più distanti dai positivi rispetto ai safe fix-paired, producendo stime meno ottimistiche e più realistiche. I safe sono disponibili separatamente per C (`c/`) e C++ (`cc/`).

La strategia consigliata è usare **solo i safe di CrossVul** come sorgente di negativi, scartando i positivi per l'eccessivo rumore.

**Quando usarlo**: esclusivamente come sorgente di safe puri da codice reale, in particolare per CWE dove gli altri dataset forniscono solo safe fix-paired.

---

#### SVEN ⚠️ secondario

Dataset multi-linguaggio (C e Python tra gli altri) con coppie funzione vulnerabile/safe estratte da commit di security fix su GitHub. CWE disponibile nella colonna `vul_type`. Label quality discreta. I safe sono fix-paired (`func_src_after`). Non ha la copertura di CWE di DiverseVul né la qualità di labeling di ICVul.

**Quando usarlo**: integrazione per aumentare i campioni su CWE specifiche dove gli altri dataset sono scarsi.

---

#### BigVul ❌ — escluso

Con label accuracy stimata al 25%, circa tre quarti dei campioni etichettati come vulnerabili potrebbero non esserlo. È il peggior valore tra tutti i dataset reali analizzati. Stimare la confidence dei SAT su un ground truth con questo livello di rumore produrrebbe stime inutilizzabili per la DST. Molti commit inclusi da BigVul in Chromium e Android non sono realmente security fix.

**Motivo di esclusione**: label accuracy insufficiente (25%), rumore non recuperabile senza filtraggio manuale esteso.

---

### 🔵 Synth

#### Juliet C/C++ 1.3 ✅ primario

Dataset sintetico NIST con 118 CWE. Ogni test case contiene esattamente una vulnerabilità di un tipo CWE specifico; le funzioni vulnerabili terminano con `_bad`, le safe con `_good`. La label accuracy è perfetta per costruzione, e i safe sono puri per design — non correlati strutturalmente ai positivi.

Il limite del branch Synth è la distanza dal codice reale: pattern semplici e ripetitivi, variabili con nomi standardizzati. I SAT hanno performance artificialmente alte su Juliet perché i pattern sono esattamente quelli per cui sono stati progettati. Le stime di confidence del branch Synth non sono direttamente comparabili con quelle del branch Real, ma costituiscono una baseline controllata preziosa. Attenzione al rischio di data leakage se si fa uno split casuale: i near-duplicate sistemici (stesso test case con variazioni minime di tipo) richiedono uno split per test case, non per funzione.

**Quando usarlo**: fonte primaria del branch Synth per C/C++. Utile anche come sorgente di safe puri per il branch Real su CWE rare.

---

#### CASTLE ✅ primario

Dataset sintetico progettato specificamente per il benchmarking di SAT e LLM su CWE in C. Ogni programma è compilabile, single-file, e contiene esattamente una o zero vulnerabilità. La qualità delle label è alta per costruzione, e la presenza di versioni safe (`"vulnerable": false`) lo rende utilizzabile come dataset completo nel branch Synth — a differenza di quanto indicato in precedenza, i campioni safe sono presenti per design in ogni entry (ogni elemento ha sia versioni vulnerabili che la versione safe).

Rispetto a Juliet, CASTLE è più recente, progettato con consapevolezza dei moderni SAT e LLM, e ha una struttura più compatta e leggibile (JSON con campo `code`, `cwe`, `vulnerable`, `lines`).

**Quando usarlo**: fonte primaria del branch Synth, in combinazione con Juliet. Particolarmente adatto come dataset di validazione per la sua struttura controllata.

---

### 🟡 AI

#### LLMSecEval ✅ primario

Codice C generato da GitHub Copilot per scenari CWE specifici. I safe sono disponibili come "Secure Code Samples" separati — generati da LLM come versione sicura — quindi sono puri nel contesto AI (non sono fix del vulnerabile ma alternative generate ex novo). CWE disponibile tramite struttura a cartelle. La label quality è discreta: il labeling è manuale o semi-manuale sulla base degli scenari di generazione, non tramite SAT, il che evita il circolo vizioso.

Non tutti i campioni sono C (alcuni sono Python): la selezione corretta richiede il filtraggio per estensione `.c`.

**Quando usarlo**: fonte primaria del branch AI per C. Copre un sottoinsieme di CWE rilevanti per il codice di sistema.

---

#### FormAI ⚠️ secondario

Dataset di 112.000 programmi C compilabili generati da LLM, labellati tramite ESBMC (Efficient SMT-based Context-Bounded Model Checker). L'uso di un model checker formale — invece di SAT tradizionali — evita il circolo vizioso di Draper VDISC e D2A: ESBMC ha caratteristiche molto diverse dagli strumenti di analisi statica che si vuole combinare con DST, quindi le sue label non sono sistematicamente distorte verso i pattern di quei tool.

Il limite principale è la copertura: ESBMC rileva con alta accuratezza vulnerabilità di memoria (buffer overflow, array bound violations, null dereference) ma non può identificare CWE come injection o XSS. La colonna CWE è parziale di conseguenza.

**Quando usarlo**: fonte secondaria del branch AI per C, in particolare per CWE di memoria. Verificare la disponibilità e struttura della colonna CWE prima dell'integrazione.

---
---

# Java

## Tabella riepilogativa

| Dataset | Branch | CWE | Vuln | Safe | Tipo safe | Label quality | Giudizio |
|---|---|---|---|---|---|---|---|
| **CVEfixes** | 🟢 Real | ✅ (via zenodo) | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐ | ✅ primario |
| **ReposVul** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐⭐ | ✅ primario |
| **CWE-Bench-Java** | 🟢 Real | ✅ (4 CWE) | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐⭐ | ✅ primario (4 CWE) |
| **CrossVul** | 🟢 Real | ✅ | ✅ | ✅ | **puri** | ⭐⭐ (48%) | ⚠️ solo safe |
| **Vul4J** | 🟢 Real | ✅ | ✅ | ❌ (esterni) | n/a | ⭐⭐⭐ | ⚠️ secondario |
| **Juliet Java 1.3** | 🔵 Synth | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐⭐⭐ | ✅ primario |
| **OWASP Benchmark** | 🔵 Synth | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐⭐⭐ | ✅ primario |
| **SARD Java** | 🔵 Synth | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐⭐⭐ | ⚠️ secondario |
| **CAPEC_LLM** | 🟡 AI | ⚠️ filtrabile | ✅ | ❌ (esterni) | n/a | ⭐⭐ | ⚠️ secondario |

---

## Analisi critica per dataset

### 🟢 Real

#### CVEfixes ✅ primario

Stesse caratteristiche della versione C/C++. Per Java, il sottoinsieme si estrae filtrando `language == 'java'`. La CWE richiede join con NVD dalla versione zenodo originale; la versione Kaggle va evitata per uso per-CWE. I safe sono fix-paired.

**Quando usarlo**: fonte primaria del branch Real per Java, in combinazione con ReposVul.

---

#### ReposVul ✅ primario

Stesse caratteristiche della versione C/C++. Per Java, il sottoinsieme è scaricabile separatamente. Il vulnerability untangling module riduce significativamente il problema delle funzioni irrilevanti etichettate come vulnerabili, rendendolo metodologicamente superiore a CVEfixes per label quality. La copertura di 236 CWE e la validazione manuale lo rendono la migliore aggiunta recente per il branch Real Java.

**Quando usarlo**: fonte primaria del branch Real per Java. Da preferire a CVEfixes come prima scelta per label quality; usare CVEfixes come complemento per CWE con pochi campioni in ReposVul.

---

#### CWE-Bench-Java ✅ primario (4 CWE)

Dataset di 120 CVE in progetti Java reali, compilabili, con labeling manuale. Copre esattamente quattro CWE: CWE-022 (path traversal), CWE-078 (OS command injection), CWE-079 (XSS), CWE-094 (code injection). Ogni CVE include il codice sorgente vulnerabile e fixed del progetto intero, con informazioni sulle funzioni e classi coinvolte nel fix — molte delle quali manualmente verificate. I progetti sono complessi (media 300K LOC), il che lo rende il dataset più realistico disponibile per Java su queste CWE.

I safe sono il codice post-fix del progetto, recuperabili tramite gli script forniti. Come Vul4J, richiede un passaggio di estrazione, ma la qualità del labeling giustifica il costo operativo. Non copre CWE al di fuori delle quattro indicate.

**Quando usarlo**: gold standard per le CWE-022, CWE-078, CWE-079, CWE-094 in Java. Per queste quattro CWE specifiche, da usare come dataset di riferimento primario e verificare la coerenza delle stime ottenute sugli altri dataset.

---

#### CrossVul ⚠️ solo safe

Stessa analisi della versione C/C++: safe puri da codice reale (sottocartella `java/`), positivi da scartare per label accuracy al 48%. La struttura `bad_X` / `good_X` garantisce safe strutturalmente indipendenti dai positivi.

**Quando usarlo**: esclusivamente come sorgente di safe puri da codice reale Java, per CWE dove CVEfixes fornisce solo safe fix-paired.

---

#### Vul4J ⚠️ secondario

Dataset Java basato su commit reali con tracciabilità del fix. La CWE è disponibile, la label quality è discreta. Il codice non è incluso direttamente nel repository ma va estratto dalle repository originali tramite lo script fornito dagli autori — un costo operativo non trascurabile, con dipendenza dalla disponibilità online delle repo al momento dell'estrazione. Non contiene campioni safe, che vanno recuperati da CVEfixes, CrossVul o Juliet.

**Quando usarlo**: integrazione per aumentare i positivi reali Java su CWE specifiche, accettando il costo operativo di estrazione. Verificare la disponibilità delle repo sorgente prima di investire nel preprocessing.

---

### 🔵 Synth

#### Juliet Java 1.3 ✅ primario

Dataset sintetico NIST con 112 CWE per Java. Struttura analoga alla versione C/C++: funzioni vulnerabili chiamate `bad`, safe chiamate `good` o `goodXxx` (possono essere più di una per test case). Label accuracy perfetta, safe puri per design. La CWE va estratta dal package name (es. `testcases.CWE113_HTTP_Response_Splitting`) o dai commenti iniziali del file.

Stessa avvertenza sulla distanza dal codice reale e sul rischio di data leakage con split casuale.

**Quando usarlo**: fonte primaria del branch Synth per Java. Copertura ampia di CWE (112 tipi).

---

#### OWASP Benchmark Java ✅ primario

Dataset sintetico progettato specificamente per il benchmarking di SAT su applicazioni Java. La mappatura è in un CSV con colonne `test name`, `real vulnerability` (true/false), e `cwe`. Il codice sorgente è in file `.java` referenziati dal test name. Safe puri per costruzione. Copre CWE tipiche delle applicazioni web Java: injection (SQL, OS, LDAP, XPath), path traversal, XSS, weak crypto, cookie security, ecc.

Rispetto a Juliet, OWASP Benchmark è progettato con focus esplicito sui SAT — gli autori lo usano per misurare TPR e FPR degli strumenti di analisi statica — il che lo rende particolarmente adatto al tuo obiettivo. La struttura del codice è più vicina ad applicazioni reali rispetto a Juliet, pur rimanendo sintetica.

**Quando usarlo**: fonte primaria del branch Synth per Java, in combinazione con Juliet. Particolarmente indicato per la stima della confidence su CWE web-related.

---

#### SARD Java ⚠️ secondario

Il SARD (Software Assurance Reference Dataset) del NIST è la suite madre da cui deriva Juliet, ma contiene test case Java anche al di fuori del subset Juliet 1.3 — inclusi contributi della comunità e altre suite NIST. Copre 150 CWE per Java (contro le 112 di Juliet Java 1.3), con 29.258 campioni safe e 12.954 vulnerabili. La struttura e le garanzie di label quality sono analoghe a Juliet: label perfette per costruzione, safe puri per design.

La differenza rispetto a Juliet è la maggiore eterogeneità dei test case: alcuni provengono da suite diverse con stili di codice leggermente diversi, il che può ridurre marginalmente la ripetitività dei pattern. Vale la pena interrogarlo per CWE presenti in SARD ma non in Juliet Java 1.3.

**Quando usarlo**: complemento a Juliet Java 1.3 per CWE non coperte da quest'ultimo. Da trattare con le stesse cautele di Juliet riguardo al distribution shift rispetto al codice reale.

---

### 🟡 AI

Dataset di codice Java (e altri linguaggi) generato da LLM per scenari di attacco CAPEC. Le CWE non sono in una colonna strutturata ma vanno estratte con parsing dal campo `description` (es. "CWE-120", "CWE-20"): i campioni in cui la CWE non è parsabile vanno scartati, ma quelli in cui è presente sono utilizzabili. Non contiene campioni safe, recuperabili da Juliet o OWASP Benchmark. La label quality è limitata dalla natura AI-generated senza labeling indipendente dei positivi.

**Quando usarlo**: integrazione nel branch AI per Java per i campioni con CWE parsabile dal campo `description`. Richiede uno step di parsing e filtraggio come preprocessing obbligatorio. È l'unica fonte disponibile per il branch AI Java, il che ne giustifica l'inclusione nonostante i limiti.

---
---

# Python

## Tabella riepilogativa

| Dataset | Branch | CWE | Vuln | Safe | Tipo safe | Label quality | Giudizio |
|---|---|---|---|---|---|---|---|
| **CVEfixes** | 🟢 Real | ✅ (via zenodo) | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐ | ✅ primario |
| **ReposVul** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐⭐ | ✅ primario |
| **PatchEval** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐⭐⭐ | ✅ primario |
| **CrossVul** | 🟢 Real | ✅ | ✅ | ✅ | **puri** | ⭐⭐ (48%) | ⚠️ solo safe |
| **PyVul** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐ | ⚠️ secondario |
| **SVEN** | 🟢 Real | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐ | ⚠️ secondario |
| **OWASP Benchmark** | 🔵 Synth | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐⭐⭐ | ✅ primario |
| **LLMSecEval** | 🟡 AI | ✅ | ✅ | ✅ | **puri** | ⭐⭐⭐ | ✅ primario |
| **synth-vuln-fixes** | 🟡 AI | ✅ | ✅ | ✅ | fix-paired | ⭐⭐⭐ | ⚠️ secondario |
| **SecurityEval** | 🟡 AI | ✅ | ✅ | ❌ (esterni) | n/a | ⭐⭐⭐ | ⚠️ secondario |
| **CAPEC_LLM** | 🟡 AI | ⚠️ filtrabile | ✅ | ❌ (esterni) | n/a | ⭐⭐ | ⚠️ secondario |
| **PythonVulnDB** | 🔵 Synth | ✅ | ✅ | ✅ | puri | ❌ (GPT-4+SAT) | ❌ |

---

## Analisi critica per dataset

### 🟢 Real

#### CVEfixes ✅ primario

Stesse caratteristiche degli altri linguaggi. Per Python, il sottoinsieme si estrae filtrando `language == 'py'`. La CWE richiede join con NVD dalla versione zenodo; la versione Kaggle va evitata. I safe sono fix-paired.

**Quando usarlo**: fonte primaria del branch Real per Python, in combinazione con ReposVul e PatchEval.

---

#### ReposVul ✅ primario

Stesse caratteristiche della versione C/C++ e Java. Per Python, il sottoinsieme è scaricabile separatamente. Il vulnerability untangling module garantisce label quality superiore rispetto a CVEfixes. Copre 236 CWE con validazione manuale.

**Quando usarlo**: fonte primaria del branch Real per Python. Da preferire a CVEfixes come prima scelta per label quality.

---

#### PatchEval ✅ primario

Dataset ByteDance (2025) di 1.000 CVE reali con labeling manuale rigoroso, che copre 65 CWE su Python, Go e JavaScript. La struttura include `cve_id`, `cwe_info` (dizionario strutturato con dettagli CWE), `vul_func` (funzioni vulnerabili), `fix_func` (funzioni fixed), e `vul_patch` (diff). Un sottoinsieme di 230 CVE è corredato da ambienti Docker per la validazione runtime tramite PoC e unit test — il che rappresenta il livello più alto di verifica disponibile in qualsiasi dataset pubblico. I CVE coprono il periodo 2015–2025. Il labeling include una fase di disentangling manuale dei commit, analoga all'approccio di ReposVul ma con validazione eseguibile.

Il dataset copre solo Python tra i linguaggi del nostro studio (Go e JavaScript sono fuori scope). I safe sono fix-paired (`fix_func`). La CWE è disponibile come campo strutturato `cwe_info`, non richiede join esterni.

**Quando usarlo**: fonte primaria di altissima qualità per il branch Real Python. Per i 230 CVE con ambiente Docker, usarlo come gold standard per la validazione delle stime di confidence — le label sono verificabili a runtime. Da trattare come il dataset più affidabile disponibile per Python.

---

#### CrossVul ⚠️ solo safe

Stessa analisi degli altri linguaggi: safe puri da codice reale (sottocartella `py/`), positivi da scartare per label accuracy al 48%.

**Quando usarlo**: esclusivamente come sorgente di safe puri da codice reale Python.

---

#### PyVul ⚠️ secondario

Dataset Python con coppie vulnerabile/safe in formato `.jsonl` per CWE. La label si trova nella sezione `assistant` del formato conversazionale (`True` = vulnerabile, `False` = safe). La struttura per-CWE (un file jsonl per CWE) è comoda per l'analisi per-CWE. I safe sono fix-paired. La label quality è discreta; la struttura conversazionale richiede un parsing dedicato.

**Quando usarlo**: integrazione per aumentare i campioni Python su CWE specifiche. Verificare la label quality su un campione manuale prima di includerlo nella stima.

---

#### SVEN ⚠️ secondario

Dataset multi-linguaggio con coppie vuln/safe, CWE in `vul_type`, safe fix-paired. Per Python si filtra su `file_name` con estensione `.py`. Label quality discreta. Non ha la copertura di CWE di DiverseVul né la qualità di labeling dei dataset primari.

**Quando usarlo**: integrazione per CWE Python con pochi campioni in CVEfixes e PyVul.

---

### 🔵 Synth

#### OWASP Benchmark Python ✅ primario

Versione Python di OWASP Benchmark, con struttura identica alla versione Java: CSV con mapping test case → `real vulnerability` → `cwe`, codice sorgente in file `.py`. Safe puri per costruzione, CWE esplicita, label perfetta. Copre CWE tipiche delle applicazioni web Python: injection, path traversal, XSS, weak crypto, ecc.

Come per Java, la progettazione esplicita per il benchmarking di SAT lo rende particolarmente adatto alla stima della confidence. Il limite è la natura sintetica del codice, lontana dai pattern del codice Python reale.

**Quando usarlo**: fonte primaria del branch Synth per Python. Particolarmente indicato per CWE web-related.

---

#### PythonVulnDB ❌ — escluso

Dataset sintetico Python con safe puri. Il labeling è stato effettuato tramite **GPT-4 e tool di analisi statica**. Questo introduce un doppio circolo vizioso: le label dipendono da un LLM (non verificabile e non riproducibile) e dagli stessi SAT che si vuole valutare. Le stime di confidence calcolate su questo ground truth sarebbero sistematicamente distorte verso i pattern visibili a quei tool specifici.

**Motivo di esclusione**: labeling tramite SAT — circolo vizioso diretto con l'obiettivo della ricerca. L'esclusione è indipendente dalla natura sintetica del codice.

---

### 🟡 AI

#### LLMSecEval ✅ primario

Codice Python e C generato da GitHub Copilot per scenari CWE specifici. I safe sono disponibili come "Secure Code Samples" separati — generati da LLM come versione sicura, quindi **puri nel contesto AI** (non fix del vulnerabile ma alternative generate ex novo). CWE disponibile tramite struttura a cartelle. Il labeling è basato sugli scenari di generazione, non su SAT, evitando il circolo vizioso.

Per il branch AI Python, filtrare i campioni con estensione `.py`. La selezione corretta richiede attenzione perché il dataset mescola C e Python nello stesso archivio.

**Quando usarlo**: fonte primaria del branch AI per Python (e C). Dataset AI con safe puri e CWE strutturata.

---

#### synth-vuln-fixes ⚠️ secondario

Dataset di vulnerabilità Python generate da GPT-4 e validate tramite human review e static analysis, disponibile su Hugging Face (patched-codes/synth-vuln-fixes). La presenza del human review lo distingue da PythonVulnDB — il circolo vizioso con i SAT è parziale, non totale, poiché la validazione umana introduce un livello di verifica indipendente. Contiene coppie vulnerabile/fix con CWE associata. I safe sono fix-paired (versione fixata del codice vulnerabile).

Usato in letteratura recente come benchmark di validazione indipendente per modelli Python, a conferma della sua utilità come dataset di test. La componente SAT nel processo di validazione rimane un rischio da tenere presente quando si stimano le confidence di quei tool specifici.

**Quando usarlo**: integrazione nel branch AI per Python come complemento a LLMSecEval, in particolare per CWE non coperte da quest'ultimo. Verificare quali SAT sono stati usati nella validazione per evitare bias nella stima della loro confidence.

---

#### SecurityEval ⚠️ secondario

Dataset di codice Python vulnerabile AI-generated (Copilot), organizzato per CWE in sottocartelle — la CWE è quindi disponibile in modo diretto e strutturato dalla struttura delle directory. Non contiene campioni safe, che vanno recuperati da OWASP Benchmark o CrossVul. La label quality è discreta: il labeling è basato sugli scenari di generazione per CWE, senza dipendenza da SAT.

**Quando usarlo**: integrazione nel branch AI per Python, in particolare per le CWE coperte da Copilot negli scenari di generazione. Da affiancare a LLMSecEval per ampliare la copertura di CWE nel branch AI.

---

#### CAPEC_LLM ⚠️ secondario

Stessa analisi della versione Java: CWE estraibile con parsing dal campo `description`, campioni senza CWE parsabile da scartare. Nessun campione safe, recuperabili da OWASP Benchmark o CrossVul. Filtrare per estensione `.py` per il sottoinsieme Python.

**Quando usarlo**: integrazione nel branch AI per Python per i campioni con CWE parsabile. Preprocessing obbligatorio per estrazione CWE e filtraggio per linguaggio.

---
---

# Riepilogo per branch

## Branch Real 🟢

| Linguaggio | Dataset primari | Dataset secondari | Solo safe |
|---|---|---|---|
| **C/C++** | ICVul, ReposVul, MegaVul, CVEfixes, PrimeVul | DiverseVul, SVEN | CrossVul |
| **Java** | CVEfixes, ReposVul, CWE-Bench-Java* | Vul4J | CrossVul |
| **Python** | CVEfixes, ReposVul, PatchEval | PyVul, SVEN | CrossVul |

*CWE-Bench-Java è primario solo per le 4 CWE coperte (CWE-022, CWE-078, CWE-079, CWE-094).

L'aggiunta di ReposVul risolve in parte il gap tra C/C++ e gli altri linguaggi: Java e Python dispongono ora di una fonte primaria di alta qualità con vulnerability untangling oltre a CVEfixes. PatchEval aggiunge per Python un dataset con validazione runtime che non ha equivalenti negli altri linguaggi. Il gap rimane significativo per il branch Real Java, dove CWE-Bench-Java copre solo 4 CWE.

## Branch Synth 🔵

| Linguaggio | Dataset primari | Note |
|---|---|---|
| **C/C++** | Juliet C/C++ 1.3, CASTLE | Attenzione al data leakage con split casuale su Juliet |
| **Java** | Juliet Java 1.3, OWASP Benchmark | SARD Java come secondario per CWE non coperte da Juliet |
| **Python** | OWASP Benchmark | Nessun equivalente di Juliet per Python |

## Branch AI 🟡

| Linguaggio | Dataset primari | Dataset secondari | Note |
|---|---|---|---|
| **C** | LLMSecEval | FormAI | FormAI limitato alle CWE di memoria |
| **Java** | — | CAPEC_LLM | Solo campioni con CWE parsabile da `description`; safe da recuperare |
| **Python** | LLMSecEval | synth-vuln-fixes, SecurityEval, CAPEC_LLM | Verificare SAT usati in synth-vuln-fixes; safe da recuperare per SecurityEval e CAPEC_LLM |

Il branch AI per Java rimane il più povero: CAPEC_LLM è l'unica fonte disponibile con qualità limitata e CWE non strutturata. Per tutti i branch AI è consigliabile tenere traccia attiva della letteratura recente, dove nuovi dataset AI-generated per il benchmarking della security stanno emergendo rapidamente.

---

# Criteri di esclusione applicati

I dataset sono stati esclusi esclusivamente per i seguenti motivi, indipendentemente dall'origine del codice. L'assenza di campioni safe e la CWE non strutturata non sono criteri di esclusione autonomi se il dato è recuperabile o filtrabile.

| Motivo | Dataset esclusi |
|---|---|
| Label accuracy insufficiente | BigVul (25%) |
| Circolo vizioso nel labeling (SAT + LLM) | PythonVulnDB |
