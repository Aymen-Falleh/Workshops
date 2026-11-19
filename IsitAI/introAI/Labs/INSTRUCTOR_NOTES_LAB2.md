# Lab 2 · Find the weird transactions — Instructor Notes

**Fundamentals of AI · 2-hour workshop · ~22 min slot**
Notebook: `lab2_find_the_weird_transactions.ipynb` (32 cells, runs top-to-bottom in Colab, no login).
Same dataset as Lab 1 — labels are locked in `answer_key` until §6 to simulate "no labels yet".

## Colab upload steps

1. Colab → **File → Upload notebook** → `lab2_find_the_weird_transactions.ipynb`. CPU runtime is fine.
2. **Runtime → Run all**. First run downloads ~150 MB via `fetch_openml(data_id=1597)`; if it fails, the load cell prints the Kaggle-upload fallback (`creditcard.csv` next to the notebook also works).
3. Two blanks, each followed by a collapsed **"Show solution"** cell. Blank cells are plain editable cells (not form mode). `#@param` sliders (k, use_scaling, flag %) are the knobs — students move them and re-run that cell.
4. `<details>` checkpoint blocks render fine in Colab markdown (same mechanism as Lab 1).

## Expected outputs (measured on a real nbconvert run, sklearn 1.9.1 / pandas 3.0.1)

| Step | Expected result |
|---|---|
| Load | 284,807 transactions; `answer_key` holds 492 frauds, hidden until §6 |
| Blank #1 | `X_scaled` — every feature mean ≈ 0, std ≈ 1 |
| K-Means k=4, scaling ON (50k sample) | cluster sizes ≈ [18424, 21275, 6806, 3495] |
| Elbow plot | inertia falls steeply to ~k=4–6 then flattens |
| PCA 2D | **2 components keep ~11.1%** of the information; clusters visible as blobs |
| Isolation Forest, flag top 1% | **2,849 flagged** |
| Reveal (Blank #2) | **303 of 2,849 flags are truly fraud** (of 492 total) |
| Scorecard | precision **0.106** · recall **0.616** · base rate 0.173% · **lift ≈ 61.6×** |
| Fraud share per cluster | cluster 3 ≈ **0.625%** fraud (~3.6× base), cluster 2 ≈ 0.013% — clusters differ in fraud density |
| Stretch (30k stratified sample) | IF: precision 0.090, recall 0.519, lift 51.9× · LOF: precision 0.017, recall 0.096, lift 9.6× |

Notes for the instructor:
- **Uncheck "use scaled features"** and re-run 3a/4/5: `Amount` dominates distances — clusters become basically "small vs large amount". Great 30-second demo of why §2 matters.
- The lift number is the money shot: flagging 1% of transactions surfaces a group ~62× richer in fraud than average — but still only ~1 in 9 flags is real fraud. That's why humans review.
- LOF honestly underperforms IF here (lift ~10× vs ~52× on the same sample) — worth a sentence: neighbor-based methods struggle when anomalies are scattered and the feature space is 30-dim.
- Numbers are stable (`random_state` fixed); kmeans cluster sizes vary a little if students change k — that's the point of the knob.

## Checkpoint answers (also in collapsed `<details>` blocks)

- **§2 — why scale before K-Means?** K-Means uses distances; unscaled `Amount` (0–25k) dwarfs `V1`–`V28` (−10–10), so only Amount would matter.
- **§3 — read the elbow** inertia = within-cluster spread; always decreases with k. Elbow = where gains flatten; past it you're splitting noise.
- **§4 — what is PCA?** Finds the 2 directions of maximum spread and projects onto them — a picture of 30 dims, at the cost of losing detail.
- **§5 — what does Isolation Forest assume?** Anomalies are few and different → isolated by random splits in fewer steps. It finds oddballs, not fraud.
- **§6 — the lesson** K-Means groups, IF finds outliers; neither knows what fraud is. Humans must review flags — precision = analyst workload.

## Blank solutions

- **Blank #1** — `X_scaled = StandardScaler().fit_transform(X_all)`
- **Blank #2** — `fraud_caught = int(answer_key[flagged].sum())`

## Runtime & verification

- Executed end-to-end via `nbconvert --execute` with both blanks filled: **no errors**; all charts, the cluster table, and widgets rendered.
- Wall time a few minutes, dominated by the one-time data download/parse; compute itself is seconds-to-tens-of-seconds (K-Means/IF/PCA on 50k samples, LOF on 30k).
- `fetch_openml` API verified; the 1597 download is bandwidth-limited on some networks (openml.org throttling observed locally). Colab is normally fast; fallback covers slow connections.
