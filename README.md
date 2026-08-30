# numerical-methods

Progetto d'esame per un corso di metodi numerici: 5 notebook, ciascuno costruito a partire da una tecnica vista a lezione ma reimplementata con funzioni, esempi e dati propri.

## Notebook (`notebooks/`)

- **`bezier_spoiler.ipynb`** — Curve di Bézier cubiche applicate al profilo di uno spoiler automobilistico: proprietà di interpolazione/tangenza, analisi della curvatura, visualizzazione qualitativa del flusso potenziale attorno a un cilindro equivalente.
- **`svd_recommender.ipynb`** — Fattorizzazione ai valori singolari (SVD) troncata per un sistema di raccomandazione: completamento di una matrice utenti-film con voti mancanti, scelta del rango tramite i valori singolari.
- **`least_squares.ipynb`** — Minimi quadrati per un fit di dati sperimentali: risoluzione via QR, aggiornamento della fattorizzazione con rotazioni di Givens, numero di condizionamento, regressione sulle componenti principali (PCR) in presenza di collinearità.
- **`newton_vett.ipynb`** — Metodo di Newton scalare e per sistemi non lineari: convergenza quadratica nel caso regolare, radice multipla, sensibilità al punto iniziale, Jacobiano singolare, condizionamento.
- **`newton_min.ipynb`** — Metodo di Newton smorzato per l'ottimizzazione: convergenza a un minimo nel caso regolare, punto di sella (Hessiana indefinita), effetto del fattore di smorzamento $\alpha$.

## Struttura

- `notebooks/` — i 5 notebook consegnati (sopra).
- `src/` — le funzioni riutilizzabili factored fuori dai notebook, un modulo per argomento (`bezier.py`, `svd_tools.py`, `least_squares.py`, `symbolic_utils.py`, `newton.py`).
- `plans/` — documentazione di processo (un piano per notebook).
- `codici/` — materiale di riferimento del corso, non consegnato.

## Esecuzione

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/<nome>.ipynb
```

Ogni notebook importa da `src/` aggiungendo la radice del progetto al `sys.path` nella prima cella; i dati sono generati sinteticamente (seed fisso) direttamente nei notebook, senza dipendenze da file esterni.