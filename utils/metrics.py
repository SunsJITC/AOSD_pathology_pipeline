import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def c_index(times, risks, events):
    # lifelines 约定：risk 越大生存越差；为了与 earlier event 对齐，使用 -risk
    return concordance_index(times, -risks, events)


def km_plot_by_cutoff(times, events, risks, cutoff, title, out_path):
    km = KaplanMeierFitter()
    hi = risks >= cutoff
    lo = ~hi
    fig, ax = plt.subplots(figsize=(5,4), dpi=150)
    km.fit(times[lo], events[lo], label=f"Low (n={lo.sum()})"); km.plot(ax=ax)
    km.fit(times[hi], events[hi], label=f"High (n={hi.sum()})"); km.plot(ax=ax)
    ax.set_title(title); ax.set_xlabel("Time (days)"); ax.set_ylabel("Survival")
    ax.grid(True, alpha=0.3)
    lr = logrank_test(times[hi], times[lo], events[hi], events[lo])
    ax.text(0.02, 0.02, f"Log-rank p={lr.p_value:.3g}", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def find_cutoff_max_logrank(times, events, risks, min_prop=0.2, grid=200):
    # 候选阈值位于分位数区间 [min_prop, 1 - min_prop]
    qs = np.linspace(min_prop, 1-min_prop, grid)
    ths = np.quantile(risks, qs)
    best_p = 1.0; best_th = np.median(risks)
    for th in ths:
        hi = risks >= th; lo = ~hi
        if hi.sum() < 5 or lo.sum() < 5:
            continue
        lr = logrank_test(times[hi], times[lo], events[hi], events[lo])
        if lr.p_value < best_p:
            best_p = lr.p_value; best_th = th
    return float(best_th), float(best_p)


def time_dependent_roc(times, events, risks, t):
    """Cumulative/dynamic ROC at time t (cases: event by t; controls: event-free past t)."""
    times = np.asarray(times)
    events = np.asarray(events).astype(int)
    risks = np.asarray(risks)
    # exclude censored before t
    valid = ~((events == 0) & (times <= t))
    times = times[valid]; events = events[valid]; risks = risks[valid]
    cases = (events == 1) & (times <= t)
    controls = times > t
    if cases.sum() == 0 or controls.sum() == 0:
        return None, None, np.nan
    # compute ROC by sorting unique risk thresholds
    thr = np.unique(risks)[::-1]
    tpr = []
    fpr = []
    for th in thr:
        pred = risks >= th
        tp = (pred & cases).sum()
        fp = (pred & controls).sum()
        tpr.append(tp / cases.sum())
        fpr.append(fp / controls.sum())
    # AUC (trapezoid)
    auc = np.trapz(tpr, fpr)
    return np.array(fpr), np.array(tpr), float(auc)


def time_roc_plot(splits, t, out_path, title="Time-dependent ROC"):
    """splits: list of dicts with keys name, times, events, risks."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    plotted = False
    for s in splits:
        fpr, tpr, auc = time_dependent_roc(s["times"], s["events"], s["risks"], t)
        if fpr is None:
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4)
            ax.text(0.02, 0.9, f"{s['name']}: insufficient data", transform=ax.transAxes)
            continue
        ax.plot(fpr, tpr, label=f"{s['name']} AUC={auc:.3f}")
        plotted = True
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4)
    ax.set_title(title + f" (t={t})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if plotted:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
