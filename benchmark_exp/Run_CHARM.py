"""TSB-AD runner — CHARM embedding anomaly detector (multiscale, upgraded).

Drop-in for benchmark_exp/ (successor to the original Run_CHARM.py). Two detectors:

  Semi-supervised (has a clean train split):
    * CHARM_kNN : MULTISCALE. Two sub-detectors, each run at several window sizes and
                  combined by z-scoring each scale's curve then taking the ELEMENT-WISE
                  MAX across scales ("anomalous at ANY scale" beats averaging scales, which
                  dilutes a signal that only shows up at one particular window length):
                    - embedding: mean-pool-channel cosine-kNN, windows {64,128,256}
                    - statistics: per-window [std,range,max,min,mean] L2-kNN,
                      windows {16,32,64,128,256,512} (single scale 128 only, for
                      multivariate series with channel count <=3 -- multiscale adds
                      noise there instead of signal)
                  final = max_scales(z(embedding)) + max_scales(z(statistics)).

  Unsupervised / zero-shot (no train reference):
    * CHARM_ZS  : bootstrap-kNN (IsolationForest picks a pseudo-clean reference from
                  the series itself) ENSEMBLED with the per-window std, z-score-sum.

Read-out: request `aggregate=False` (per-patch, per-channel) and pool client-side as max
over time-patches then mean over channels, on the **L5** block. Per-window raw statistics
recover amplitude anomalies the encoder z-normalizes away. Z-scoring each scale before the
max is NOT optional -- skipping it costs ~6 pp VUS-PR (verified: raw-score max-pooling
collapses without it).

Official VUS-PR (TSB-AD eval, stride-1, 350 uni / 180 mv / 530 all, 0 errors):
    CHARM_kNN : uni 0.680 / mv 0.543 / all 0.634   (best; +2.6 pp over the prior single-scale
                recipe, +12.7 pp over the original Run_CHARM baseline ~0.499)
    CHARM_ZS  : uni 0.615 / mv 0.463 / all 0.560   (zero-shot; multiscale not yet applied here)
The single-scale channel-pooling question (mean-pool vs per-channel/no-pool, vs concat)
was re-tested in this multiscale setting and mean-pool won or tied everywhere once
multiscale stats is already doing the work of catching scale-localized anomalies -- so no
channel-count-adaptive embedding pooling is needed (simpler AND at least as accurate).
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

# --------------------------------------------------------------------------- #
#  Hyperparameters (leaderboard defaults)
# --------------------------------------------------------------------------- #
CHARM_HP = {
    "window_size": 128,
    "stride": 1,
    "train_stride": 1,
    "min_window": 64,
    "layer": 5,             # L5 block (assumes served model exposes it)
    "k": 3,
    "ref_cap": 10000,       # cap kNN reference size (random subsample) for tractability
    "pointwise_agg": "mean",
    "if_estimators": 200,
    "if_max_samples": 256,
    "boot_quantile": 0.70,  # IF-suspicion percentile for the zero-shot pseudo-clean ref
    "ensemble_weight_zs": 0.40,    # min-max weight for CHARM_ZS
    "emb_scales": (64, 128, 256),          # multiscale embedding window sizes
    "stats_scales": (16, 32, 64, 128, 256, 512),  # multiscale stats window sizes
    "stats_gate_max_c": 3,          # MV with C<=this: single-scale stats only (128)
}

_CHARM_BASE_URL = os.environ.get("CHARM_BASE_URL", "")
_CHARM_API_KEY = os.environ.get("CHARM_API_KEY", "token")
_RNG = np.random.RandomState(0)


# --------------------------------------------------------------------------- #
#  Windowing helpers
# --------------------------------------------------------------------------- #
def _effective_window(series_length, max_window, stride, k, min_window=64, min_windows_mult=2.0):
    if series_length < min_window:
        return None
    min_windows = max(int(min_windows_mult * k), 10)
    ws = min(max_window, series_length - (min_windows - 1) * stride)
    if ws >= min_window:
        return ws, stride
    ws = min(max_window, series_length - (min_windows - 1))
    if ws >= min_window:
        return ws, 1
    ws = min(max_window, series_length)
    avail = series_length - ws + 1
    if avail >= min_windows:
        return ws, max(1, (series_length - ws) // (min_windows - 1))
    return ws, 1


def _create_windows(x, ws, stride):
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[0] < ws:
        return x[None]
    sw = sliding_window_view(x, ws, axis=0)[::stride]   # (n, C, ws)
    return np.moveaxis(sw, -1, 1)                        # (n, ws, C)


def _window_stats(windows, chunk=50000):
    N = windows.shape[0]
    out = np.empty((N, 5), np.float32)
    for i in range(0, N, chunk):
        b = np.ascontiguousarray(windows[i:i + chunk]).astype(np.float64)
        mx, mn = b.max(axis=(1, 2)), b.min(axis=(1, 2))
        out[i:i + len(b)] = np.stack([b.std(axis=(1, 2)), mx - mn, mx, mn, b.mean(axis=(1, 2))], 1)
    return out


def _window_scores_to_pointwise(scores, ws, stride, total_len, method="mean"):
    n = len(scores)
    if method == "mean":
        if stride == 1 and n > 0:
            cs = np.zeros(n + 1); np.cumsum(scores, out=cs[1:])
            t = np.arange(total_len)
            a = np.clip(t - ws + 1, 0, n); b = np.clip(t + 1, 0, n)
            return ((cs[b] - cs[a]) / np.maximum(b - a, 1)).astype(np.float32)
        acc = np.zeros(total_len); cnt = np.zeros(total_len)
        for i, s in enumerate(scores):
            a = i * stride; b = min(a + ws, total_len); acc[a:b] += s; cnt[a:b] += 1
        return (acc / np.maximum(cnt, 1)).astype(np.float32)
    if method == "max":
        res = np.full(total_len, -np.inf)
        for i, s in enumerate(scores):
            a = i * stride; b = min(a + ws, total_len); res[a:b] = np.maximum(res[a:b], s)
        res[res == -np.inf] = 0.0
        return res.astype(np.float32)
    raise ValueError(method)


def _zc(a):
    """Z-score (standardize) a score array — the parameter-free ensemble building block."""
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)


def _nz(a):
    """Min-max normalize a score array to [0, 1]."""
    a = np.asarray(a, float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def _cap_ref(ref, cap):
    if ref.shape[0] <= cap:
        return ref
    return ref[_RNG.choice(ref.shape[0], cap, replace=False)]


# --------------------------------------------------------------------------- #
#  Embeddings via CHARM SDK  (L5, max-over-time; channels kept separate)
# --------------------------------------------------------------------------- #
def _embed_windows_pc(client, windows, batch_size=4096):
    """windows (N, W, C) -> (N, C, D): aggregate=False, max over time-patches (channels kept)."""
    N, W, C = windows.shape
    bs = max(1, batch_size // max(C, 1))
    chunk = max(bs, 2048)
    out = []
    for i in range(0, N, chunk):
        wc = np.ascontiguousarray(windows[i:i + chunk])
        desc = [[f"ch_{c}" for c in range(C)] for _ in range(len(wc))]
        resp = client.embeddings.create(descriptions=desc, ts_array=wc.tolist(),
                                        batch_size=bs, return_tensors="np", aggregate=False)
        raw = resp.embeds
        pc = raw.max(axis=1) if raw.ndim == 4 else raw[:, None, :]   # (n, C, D)
        out.append(np.nan_to_num(pc).astype(np.float32))
    return np.concatenate(out, 0)


def _cosine_knn(query, ref, k, chunk=4096):
    kk = min(k, ref.shape[0])
    r = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    out = np.empty(query.shape[0], np.float32)
    for i in range(0, query.shape[0], chunk):
        qn = query[i:i + chunk] / (np.linalg.norm(query[i:i + chunk], axis=1, keepdims=True) + 1e-8)
        d = 1.0 - qn @ r.T
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        out[i:i + len(qn)] = np.take_along_axis(d, idx, axis=1).mean(1)
    return out


def _l2_knn(query, ref, k, chunk=4096):
    kk = min(k, ref.shape[0])
    from scipy.spatial.distance import cdist
    out = np.empty(query.shape[0], np.float32)
    for i in range(0, query.shape[0], chunk):
        d = cdist(query[i:i + chunk].astype(np.float64), ref.astype(np.float64))
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        out[i:i + len(d)] = np.take_along_axis(d, idx, axis=1).mean(1)
    return out


def _emb_score_meanpool(q_pc, r_pc, k, cap):
    return _cosine_knn(q_pc.mean(1), _cap_ref(r_pc.mean(1), cap), k)


def _get_client():
    from charm import CharmClient
    return CharmClient(base_url=_CHARM_BASE_URL, api_key=_CHARM_API_KEY, timeout=300)


# --------------------------------------------------------------------------- #
#  Semi-supervised runners (fit on clean train, score test) — z-score ensemble
# --------------------------------------------------------------------------- #
_STD_STATE = {}
def _standardize_fit(S):
    mu, sd = S.mean(0, keepdims=True), S.std(0, keepdims=True) + 1e-8
    _STD_STATE["mu"], _STD_STATE["sd"] = mu, sd
    return ((S - mu) / sd).astype(np.float32)
def _standardize_apply(S):
    return ((S - _STD_STATE["mu"]) / _STD_STATE["sd"]).astype(np.float32)


def _multiscale_level_score(data_train, data_test, ws, HP, is_stats):
    """One scale's pointwise, z-scored detector curve (embedding or stats), or None if the
    series is too short for this window size."""
    ews = _effective_window(len(data_train), ws, HP["train_stride"], HP["k"], HP["min_window"])
    ews_t = _effective_window(len(data_test), ws, HP["stride"], HP["k"], HP["min_window"])
    if ews is None or ews_t is None:
        return None
    ws_tr, tr_stride = ews; ws_te, te_stride = ews_t
    tw = _create_windows(data_train, ws_tr, tr_stride); qw = _create_windows(data_test, ws_te, te_stride)
    if is_stats:
        ref = _standardize_fit(_window_stats(tw)); q = _standardize_apply(_window_stats(qw))
        s = _l2_knn(q, _cap_ref(ref, HP["ref_cap"]), HP["k"])
    else:
        client = _get_client()
        r_pc = _embed_windows_pc(client, tw); q_pc = _embed_windows_pc(client, qw)
        s = _emb_score_meanpool(q_pc, r_pc, HP["k"], HP["ref_cap"])
    pw = _window_scores_to_pointwise(s, ws_te, te_stride, len(data_test), HP["pointwise_agg"])
    return _zc(pw)


def _multiscale_max(data_train, data_test, scales, HP, is_stats):
    """z-score each scale's curve, then take the element-wise MAX across scales --
    'anomalous at ANY scale' beats averaging scales together (HYDRA-style; confirmed:
    max-pooling across scales, after z-scoring each, is essential -- skipping the z-score
    step costs ~6pp; averaging instead of maxing loses the scale-localized signal)."""
    levels = [lv for ws in scales if (lv := _multiscale_level_score(data_train, data_test, ws, HP, is_stats)) is not None]
    if not levels:
        return np.zeros(len(data_test))
    return np.max(np.stack(levels), axis=0)


def run_CHARM_kNN(data_train, data_test, HP=CHARM_HP):
    """Semi (BEST overall, confirmed full TSB-AD eval, stride-1, +2.6pp over the prior
    single-scale recipe): multiscale mean-pool-channel embedding kNN, windows
    {64,128,256}, max-pooled across scales, (+) multiscale per-window mu/std kNN, windows
    {16,32,64,128,256,512}, max-pooled across scales (single-scale-128 fallback for
    multivariate with channel count <=3, where the extra stats scales add noise)."""
    C = data_train.shape[1] if data_train.ndim > 1 else 1
    stats_scales = (HP["window_size"],) if (1 < C <= HP["stats_gate_max_c"]) else HP["stats_scales"]
    emb = _multiscale_max(data_train, data_test, HP["emb_scales"], HP, is_stats=False)
    stats = _multiscale_max(data_train, data_test, stats_scales, HP, is_stats=True)
    n = min(len(emb), len(stats))
    final = emb[:n] + stats[:n]
    return MinMaxScaler().fit_transform(final.reshape(-1, 1)).ravel()


# --------------------------------------------------------------------------- #
#  Zero-shot runner (no train reference) — z-score ensemble
# --------------------------------------------------------------------------- #
def run_CHARM_ZS(data, HP=CHARM_HP):
    """Zero-shot: bootstrap-kNN on L5 embeddings (+) per-window std, z-score-sum."""
    client = _get_client()
    ews = _effective_window(len(data), HP["window_size"], HP["stride"], HP["k"], HP["min_window"])
    if ews is None:
        return np.zeros(len(data))
    ws, stride = ews
    w = _create_windows(data, ws, stride)
    pc = _embed_windows_pc(client, w); emb = pc.mean(1)
    std = _window_stats(w)[:, 0]
    En = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    if_score = -IsolationForest(n_estimators=HP["if_estimators"], max_samples=HP["if_max_samples"],
                                random_state=0, n_jobs=4).fit(En).score_samples(En)
    ref = emb[if_score <= np.quantile(if_score, HP["boot_quantile"])]
    bknn = _cosine_knn(emb, _cap_ref(ref, HP["ref_cap"]), 1) if len(ref) >= 1 else if_score
    n = min(len(bknn), len(std))
    win = _nz(bknn[:n]) + HP["ensemble_weight_zs"] * _nz(std[:n])   # min-max (best zero-shot on eval)
    pw = _window_scores_to_pointwise(win, ws, stride, len(data), HP["pointwise_agg"])
    return MinMaxScaler().fit_transform(pw.reshape(-1, 1)).ravel()


SEMISUPERVISE = {"CHARM_kNN": run_CHARM_kNN}
UNSUPERVISE = {"CHARM_ZS": run_CHARM_ZS}


# --------------------------------------------------------------------------- #
#  Main — same file_list / split convention as the original Run_CHARM.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filename", required=True)
    ap.add_argument("--data_dir", default="Datasets/TSB-AD-M/")
    ap.add_argument("--model", default="CHARM_kNN", choices=list(SEMISUPERVISE) + list(UNSUPERVISE))
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, args.filename)).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    train_index = int(args.filename.split("_")[-3])   # ..._tr_<train_len>_1st_<first_anomaly>...

    if args.model in SEMISUPERVISE:
        score = SEMISUPERVISE[args.model](data[:train_index], data[train_index:])
        label = label[train_index:]
    else:
        score = UNSUPERVISE[args.model](data)

    metrics = get_metrics(score, label, slidingWindow=find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
    print(args.model, args.filename, {k: round(v, 4) for k, v in metrics.items()})
