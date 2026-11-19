# Lab 1 · Catch the fraud — Instructor Notes

**Fundamentals of AI · 2-hour workshop · ~25 min slot**
Notebook: `lab1_catch_the_fraud.ipynb` (36 small cells — one idea per cell, runs top-to-bottom in Colab, no login required).

## Colab upload steps

1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → pick `lab1_catch_the_fraud.ipynb`.
2. Runtime → Change runtime type → **CPU** is fine (GPU not needed).
3. Run cells top to bottom (**Runtime → Run all**). First run downloads ~150 MB from OpenML — give it a minute.
4. The two blanks (Blank #1 stratified split, Blank #2 `class_weight="balanced"`) have a **"Show solution"** form cell right under each — students unfold it only if stuck. Both blanks degrade gracefully if left empty (friendly printed message, no traceback).

## Data

- Source: [Credit Card Fraud Detection — ULB (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), fetched automatically via `fetch_openml(data_id=1597)` — no login.
- Fallback: if the OpenML download fails, the setup cell prints instructions to upload `creditcard.csv` from Kaggle (the cell also auto-loads a `creditcard.csv` sitting next to the notebook).
- `Class` is cast to int 0/1; features are `Time`, `Amount`, `V1`–`V28`.

## Expected outputs (measured on a real run, sklearn 1.9.1 / pandas 3.0.1)

| Step | Expected result |
|---|---|
| Data load | 284,807 transactions × 30 features; **492 fraud / 284,315 legit** |
| Fraud rate | **0.173%** — about **1 fraud per 578 legit** transactions |
| Always-legit baseline | Accuracy **99.83%**, frauds caught **0 / 492** |
| Train/test split (Blank #1) | Train 227,845 (0.173% fraud) / Test 56,962 (0.172% fraud) — ratios match thanks to `stratify` |
| Logistic Regression | precision **0.827**, recall **0.633** |
| Decision tree `max_depth=10` | train acc **100.0%**, test acc **99.9%** — overfitting visible; try `max_depth=2` and `None` with the knob |
| Random Forest + `class_weight="balanced"` (Blank #2) | precision **0.698**, recall **0.827** — catches ~20 pts more fraud, more false alarms |
| Final comparison | baseline recall 0.000 · LR recall 0.633 · RF recall 0.827 |
| Cost game | best swept threshold ≈ **0.20**, total cost ≈ **$5,880** (missed frauds $500 each, false alarms $5 each) |

Numbers will wobble slightly run-to-run (RF has `random_state=42`, so they should be stable; small diffs possible across sklearn versions).

## Checkpoint answers (also hidden in `<details>` blocks in the notebook)

- **§2 — why does rarity matter?** "Always legit" scores ~99.8% accuracy while catching zero fraud. Accuracy is meaningless on imbalanced data; use precision/recall.
- **§3 — precision vs recall?** Precision = of the alarms raised, how many are real (few false alarms). Recall = of real frauds, how many caught. Fraud detection wants high recall — a missed fraud costs far more.
- **§5 — which mistake costs more?** Missed fraud (~$500) vs false alarm (~$5): missed fraud costs ~100× more → optimize recall, tolerate false alarms.
- **§7 — why isn't best threshold 0.5?** Thresholds encode costs, not just probabilities. Since misses cost 100× alarms, the cheapest threshold is lower (~0.2) — investigate more legit txns to catch more fraud.

## Blank solutions

- **Blank #1** — inside `train_test_split(...)`: `stratify=y_all`
- **Blank #2** — inside `RandomForestClassifier(...)`: `class_weight="balanced",`

## Runtime

- Verified end-to-end via `nbconvert --execute` with both blanks filled: **no errors**, all 4 charts + comparison table + ipywidgets slider rendered.
- Wall time a few minutes, dominated by the one-time ~150 MB data download and CSV parse; model training itself is seconds (RF is capped at 50 trees, `max_depth=12`, `n_jobs=-1`). Well under the ~3 min Colab budget after the download.

## Known environment note

`fetch_openml` downloads `creditcard.arff.gz` (~150 MB) from `www.openml.org`, which throttles single connections to ~50–100 KB/s from some networks (observed locally). On Colab's network it is normally fast; if a student hits a slow/stalled download, the Kaggle-upload fallback in the setup cell covers them.
