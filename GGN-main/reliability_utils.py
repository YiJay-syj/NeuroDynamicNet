
import os
import numpy as np
import matplotlib.pyplot as plt

def _topk_confidence_and_hits(y_true, proba, k=1):
    """
    y_true: (N,) int
    proba: (N, C) float in [0,1], each row sums to 1 (softmax)
    k: 1 or 2 (or any integer >=1)
    Returns:
        conf_k: (N,) sum of top-k probs per sample
        hits_k: (N,) 1 if ground-truth is in top-k predicted labels else 0
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    # indices of top-k by row
    topk_idx = np.argsort(-proba, axis=1)[:, :k]  # (N, k)
    conf_k = np.take_along_axis(proba, topk_idx, axis=1).sum(axis=1)  # (N,)
    hits_k = (topk_idx == y_true[:, None]).any(axis=1).astype(int)    # (N,)
    return conf_k, hits_k

def _reliability_bins(conf, hits, n_bins=10, binning='quantile'):
    """
    Bin by confidence and compute per-bin mean confidence and empirical accuracy.
    Returns:
        bin_conf: (B,) mean confidence per bin
        bin_acc:  (B,) mean accuracy per bin
        bin_counts: (B,) counts per bin
        ece: scalar, sum_b (n_b/N) * |acc_b - conf_b|
    """
    conf = np.asarray(conf).astype(float)
    hits = np.asarray(hits).astype(float)
    N = len(conf)
    if N == 0:
        raise ValueError("Empty inputs to reliability computation")
    # choose edges
    if binning == 'quantile':
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        # 兼容新旧 NumPy：新版本支持 method=，旧版本用 interpolation=
        try:
            edges = np.quantile(conf, qs, method='linear')  # NumPy >= 1.22
        except TypeError:
            edges = np.quantile(conf, qs, interpolation='linear')  # 老版本兼容
        # 约束到 [0,1] 并强制首尾
        edges = np.clip(edges, 0.0, 1.0)
        edges[0], edges[-1] = 0.0, 1.0
        # 如果分位点高度重复（例如所有 conf 很接近），退回等宽分箱避免空箱/NaN
        if np.unique(edges).size < 2:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # assign bins [edges[b], edges[b+1])
    # digitize returns indices in 1..n_bins for values > edges[0]
    bins = np.digitize(conf, edges[1:-1], right=False)

    bin_conf, bin_acc, bin_counts = [], [], []
    ece = 0.0
    for b in range(n_bins):
        mask = (bins == b)
        nb = int(mask.sum())
        if nb == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            bin_counts.append(0)
            continue
        mc = float(conf[mask].mean())
        ma = float(hits[mask].mean())
        bin_conf.append(mc)
        bin_acc.append(ma)
        bin_counts.append(nb)
        ece += (nb / N) * abs(ma - mc)

    return np.array(bin_conf), np.array(bin_acc), np.array(bin_counts), float(ece)

def _plot_reliability(bin_conf, bin_acc, save_path, title="Reliability", show_diagonal=True):
    ok = ~np.isnan(bin_conf)
    plt.figure(figsize=(5, 5))
    if show_diagonal:
        plt.plot([0, 1], [0, 1], '--', linewidth=1)
    plt.plot(bin_conf[ok], bin_acc[ok], marker='o')
    plt.xlabel('Predicted confidence')
    plt.ylabel('Empirical accuracy')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_top1_top2_reliability(y_true, proba, out_dir, n_bins=10, binning='quantile', prefix=''):
    """
    Convenience wrapper: writes two figures and returns ECEs.
    Saves:
        {prefix}reliability_top1.png
        {prefix}reliability_top2.png
    Returns:
        {'ece_top1': ..., 'ece_top2': ..., 'bins': n_bins, 'binning': binning}
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)

    # Top-1
    conf1, hits1 = _topk_confidence_and_hits(y_true, proba, k=1)
    bconf1, bacc1, _, ece1 = _reliability_bins(conf1, hits1, n_bins=n_bins, binning=binning)
    _plot_reliability(bconf1, bacc1, os.path.join(out_dir, f"{prefix}reliability_top1.png"),
                      title=f"Reliability (Top-1), ECE={ece1:.3f}")

    # Top-2
    k = min(2, proba.shape[1])  # guard if C==1
    conf2, hits2 = _topk_confidence_and_hits(y_true, proba, k=k)
    bconf2, bacc2, _, ece2 = _reliability_bins(conf2, hits2, n_bins=n_bins, binning=binning)
    _plot_reliability(bconf2, bacc2, os.path.join(out_dir, f"{prefix}reliability_top2.png"),
                      title=f"Reliability (Top-2), ECE={ece2:.3f}")

    return {'ece_top1': float(ece1), 'ece_top2': float(ece2), 'bins': int(n_bins), 'binning': str(binning)}
