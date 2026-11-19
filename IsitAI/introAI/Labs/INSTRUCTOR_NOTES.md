# Lab 1 · Catch the fraud — Instructor Notes

**Fundamentals of AI · 2-hour workshop · ~25 min slot**
Notebook: `lab1_catch_the_fraud.ipynb` (35 small cells — one idea per cell, runs top-to-bottom in Colab, no login required). Cell 0 runs `%pip install -q scikit-learn pandas matplotlib ipywidgets` so it works outside Colab too.

## Colab upload steps

1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → pick `lab1_catch_the_fraud.ipynb`.
2. Runtime → Change runtime type → **CPU** is fine.
3. **Runtime → Run all**. First run downloads ~150 MB from OpenML — give it a minute.
4. Two blanks, each with a **"Show solution"** form cell right under it. Both blanks degrade gracefully if left empty (friendly printed message, no traceback). A `need()` helper stops later cells with a clear message if an earlier step hasn't run.

## Data

- Source: [Credit Card Fraud Detection — ULB (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), fetched via `fetch_openml(data_id=1597)` — no login.
- Fallback: if OpenML fails, the setup cell prints instructions to upload `creditcard.csv` from Kaggle (it also auto-loads a `creditcard.csv` next to the notebook).
- `V1`–`V28` are anonymized PCA components — transformed features, not raw measurements. `Time` (seconds since first transaction) and `Amount` are the only readable ones. `Class`: 0 = legit, 1 = fraud.

## Teaching content (the "why" behind each step)

**§2 — How rare is fraud?**
Fraud is 0.173% — about 1 in 578 transactions. Emphasize: this is why a naive model looks perfect. Have students write their guess *before* running 2a/2b — most guess 5–20%, far above reality.

**§3 — The baseline trap**
`DummyClassifier(strategy="most_frequent")` always predicts the majority class. It "cheats" by seeing all the data, but that's the point: even with perfect knowledge it only learns "say legit". Accuracy 99.83%, frauds caught 0/492. Teaching line: *accuracy measures the majority class, not the thing you care about.*

**Precision vs recall (used from §5 on):**
- Precision = of the alarms raised, how many are real? (false alarms waste analyst time)
- Recall = of the real frauds, how many did we catch? (missed fraud = $500)
- Confusion matrix: rows = truth, cols = prediction. Top-right = false alarms, bottom-left = missed frauds.

**§4 — Stratified split**
`stratify=y_all` forces train and test to keep the same ~0.17% fraud ratio. Without it, a random split could (rarely) give the test set very few frauds and make evaluation noisy. With only 492 frauds, this matters.

**§5 — Logistic Regression**
Outputs P(fraud) per transaction. `StandardScaler` inside the pipeline puts `Amount`/`Time` on the same scale as `V1`–`V28` so big values can't dominate. Plain LR already beats the baseline on recall (0.633) but misses ~37% of frauds.

**§6 — Trees, forests, class weights**
- Decision tree: recursive yes/no splits on features. `max_depth` knob shows overfitting — deep tree memorizes train (100%) but test lags; shallow tree underfits. Best depth balances both.
- Random forest: 50 trees, each on a random data/feature subset, majority vote. `max_depth=12` caps memorization.
- `class_weight="balanced"` roughly multiplies the fraud class's weight by ~578× during training, so missing a fraud hurts the model's score much more. Result: recall jumps 0.633 → 0.827, precision drops 0.827 → 0.698. **Trade-off is the lesson.**

**§7 — Cost game**
The 0.5 threshold means "more likely fraud than legit" — but ignores costs. Since missed fraud ($500) ≫ false alarm ($5), optimal threshold drops to ~0.20: the system flags ~4× more legit transactions as suspects, but catches more fraud. Real systems would add a second review stage instead of hard-blocking.

## Expected outputs (measured on a real run, sklearn 1.9.1 / pandas 3.0.1)

| Step | Expected result |
|---|---|
| Data load | 284,807 transactions × 30 features; **492 fraud / 284,315 legit** |
| Fraud rate | **0.173%** — ~1 fraud per 578 legit |
| Always-legit baseline | Accuracy **99.83%**, frauds caught **0 / 492** |
| Train/test split | Train 227,845 (0.173%) / Test 56,962 (0.172%) |
| Logistic Regression | precision **0.827**, recall **0.633** |
| Decision tree `max_depth=10` | train **100.0%**, test **99.9%** — overfitting visible |
| Random Forest + `class_weight="balanced"` | precision **0.698**, recall **0.827** |
| Final table | baseline recall 0.000 · LR 0.633 · RF 0.827 |
| Cost game | best threshold ≈ **0.20**, total cost ≈ **$5,880** |

Numbers are stable (`random_state=42` everywhere); small diffs possible across sklearn versions.

## Checkpoint answers (also in collapsed `<details>` blocks)

- **§3 — why does rarity matter?** "Always legit" = ~99.8% accuracy, 0 frauds caught. Accuracy is meaningless on imbalanced data.
- **§5 — which mistake costs more?** Missed fraud (~$500) vs false alarm (~$5): missed fraud ~100× worse → optimize recall.
- **§6 — what to look for in the table?** Baseline wins accuracy, loses recall. Forest+weights catches far more fraud at the price of more false alarms.
- **§7 — why isn't best threshold 0.5?** Thresholds encode costs, not just probabilities. Misses cost 100× alarms → cheaper to flag more legit txns.

## Blank solutions

- **Blank #1** — inside `train_test_split(...)`: `stratify=y_all`
- **Blank #2** — inside `RandomForestClassifier(...)`: `class_weight="balanced",`

## Runtime & verification

- Executed end-to-end via `nbconvert --execute` with both blanks filled: **no errors**, all charts/table/slider rendered.
- Wall time a few minutes, dominated by the one-time ~150 MB download + CSV parse; training is seconds (RF: 50 trees, `max_depth=12`, `n_jobs=-1`).
- `fetch_openml` API path verified; the 1597 download itself is bandwidth-limited on some networks (~50 KB/s observed locally vs openml.org throttling). On Colab it is normally fast; the Kaggle-upload fallback covers slow connections.
