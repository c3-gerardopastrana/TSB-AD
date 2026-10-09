# PR (DRAFT): CHARM embedding anomaly detector — multiscale (best), + zero-shot

Successor to the original `Run_CHARM.py` (PR #56). Two detectors.

## Detectors (`Run_CHARM.py`)
- **`CHARM_kNN`** (semi-supervised, best overall) — **multiscale**. Two sub-detectors,
  each run at several window sizes, z-scored per scale, combined by **element-wise MAX
  across scales** ("anomalous at ANY scale" beats averaging, which dilutes a signal that
  only shows up at one window length):
  - embedding: mean-pool-channel cosine-kNN, windows **{64,128,256}**
  - statistics: per-window `[std,range,max,min,mean]` L2-kNN, windows **{16,32,64,128,256,512}**
    (single scale 128 only for multivariate with channel count ≤3 — extra scales add noise there)
  - `final = max_scales(z(embedding)) + max_scales(z(statistics))`
- **`CHARM_ZS`** (zero-shot, no train reference) — bootstrap-kNN (IsolationForest picks a
  pseudo-clean reference from the series itself) ensembled with per-window std, z-score-sum.

## Results — VUS-PR, TSB-AD eval, stride-1, official protocol (350 uni / 180 mv / 530 all, 0 errors)
| detector | uni | mv | all |
|---|---|---|---|
| **CHARM_kNN** | **0.678** | **0.539** | **0.631** |
| CHARM_ZS | 0.596 | 0.452 | 0.547 |
| *(original Run_CHARM, last-layer mean, no μ/σ)* | — | — | ~0.499 |

`CHARM_kNN` is **+2.6pp over the single-scale recipe** and **+12.7pp over the original
baseline**. Z-scoring each scale before the max is not optional — skipping it costs ~6pp
(raw-score max-pooling across scales collapses without it). A channel-count-adaptive
embedding read-out (per-channel pooling for few channels) was re-tested in this multiscale
setting and no longer helps — plain mean-pooling ties or beats it everywhere once
multiscale statistics is already catching scale-localized anomalies, so the recipe above
needs no channel-count branching at all.

## Provenance
Read-out is CHARM's L8 block (the last layer, which is what the served endpoint returns) via `aggregate=False` (client-side max-over-time, mean-over-channel
pooling). Numbers were produced with the checkpoint that will back the served `CHARM_kNN`/`CHARM_ZS`
model; serving that endpoint is the only outstanding step. Metric verified bit-exact against
TSB-AD's `get_metrics` VUS-PR.

## How to run
```
python Run_CHARM.py --filename <ds>.csv --data_dir Datasets/TSB-AD-M/ --model CHARM_kNN
# --model CHARM_ZS   (zero-shot)
```
Env: `CHARM_BASE_URL`, `CHARM_API_KEY`; `pip install c3-charm`. Note: `CHARM_kNN` calls the
embedding endpoint at 3 window sizes per series (more requests than a single-scale recipe) —
worth it for the accuracy gain, but budget for ~3x the API calls of a single-scale detector.

## Reproducibility notes
- The endpoint serves the **last (L8) block**; the numbers above were measured on L8. Earlier drafts quoted L5 (`CHARM_kNN` 0.680 / 0.543 / 0.634, `CHARM_ZS` 0.615 / 0.463 / 0.560).
- Per-series VUS-PR for all 530 series is in `benchmark_eval_results/CHARM_{uni,multi}_mergedTable_VUS-PR.csv`.
- `Run_CHARM.py` follows the TSB-AD protocol (`Run_Detector_U.py`): fit on `data[:train_len]`, score the full series, evaluate on the full labels.
- Embeddings from the endpoint match the local L8 model (cosine 1.000). Requests are capped at the server limit of 500,000 time points and retried with a smaller batch on transient server errors (e.g. GPU OOM).
- Spot-check of `Run_CHARM.py` through the API against the table (VUS-PR, API / table):
  - `CHARM_kNN`: NAB_014 0.904 / 0.904, MSL_143 0.997 / 0.997, IOPS_267 0.310 / 0.310, YAHOO_579 1.000 / 1.000, YAHOO_649 1.000 / 1.000, YAHOO_689 0.006 / 0.006, YAHOO_741 0.616 / 0.615, Genesis (mv) 0.930 / 0.930.
  - `CHARM_ZS` is less exact because the IsolationForest reference is sampled: MSL_143 0.827 / 0.843, YAHOO_741 0.567 / 0.565, YAHOO_579 1.000 / 1.000, NAB_014 0.462 / 0.399, IOPS_267 0.303 / 0.353.
