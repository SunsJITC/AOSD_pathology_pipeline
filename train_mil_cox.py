import argparse
import glob
import json
import logging
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from models.mil import build_mil
from utils.io import ensure_dir, load_config
from utils.metrics import km_plot_by_cutoff, time_dependent_roc, time_roc_plot


def safe_cindex(T, risk, E):
    """包装一下 lifelines 的 concordance_index，避免小样本时 ZeroDivisionError."""
    try:
        return concordance_index(T, risk, E)
    except ZeroDivisionError:
        return np.nan


def cox_ph_loss(risk, times, events):
    """Mini-batch Cox partial log-likelihood (higher risk -> worse survival)."""
    order = torch.argsort(times, descending=True)
    risk_ord = risk[order]
    events_ord = events[order]
    log_cum_hazard = torch.logcumsumexp(risk_ord, dim=0)
    diff = risk_ord - log_cum_hazard
    loss = -(events_ord * diff).sum() / (events_ord.sum().clamp(min=1.0))
    return loss


class BagDS(Dataset):
    def __init__(self, ids, feats_bank, clinic_df, bag_size=4000, strategy="uniform"):
        self.ids = list(ids)
        self.bank = feats_bank
        self.clinic = clinic_df.set_index("ID")
        self.bag = bag_size
        self.strategy = strategy

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        k = self.ids[i]
        f = self.bank[k]["feats"]
        N = f.shape[0]
        if self.bag is None or self.bag <= 0:
            sel = np.arange(N)
        elif N <= self.bag:
            sel = np.arange(N)
        else:
            sel = np.linspace(0, N - 1, self.bag).astype(int) if self.strategy == "uniform" else np.random.choice(N, self.bag, replace=False)
        row = self.clinic.loc[k]
        return k, f[sel], float(row["time"]), int(row["event"])


def load_bank(wd):
    feats_dir = os.path.join(wd, "feats")
    bank = {}
    for f in sorted(glob.glob(os.path.join(feats_dir, "*.npz"))):
        k = os.path.splitext(os.path.basename(f))[0]
        data = np.load(f)
        bank[k] = {
            "feats": data["feats"].astype("float32"),
            "coords": data["coords"],
            "tile_size": int(data["tile_size"]),
            "thumb_scale": float(data["thumb_scale"]),
        }
    return bank


def aggregate_by_patient(bank):
    """如果同一患者有多张切片（命名如 1234567-1），将 feats 按患者 ID 聚合."""

    def slide_to_patient(slide_id: str) -> str:
        m = re.match(r"^(.+)-(\d+)$", slide_id)
        if not m:
            return slide_id
        prefix, suffix = m.group(1), int(m.group(2))
        has_alpha = any(ch.isalpha() for ch in prefix)
        num_digits = sum(ch.isdigit() for ch in prefix)
        if suffix <= 20 and (has_alpha or num_digits >= 6):
            return prefix
        return slide_id

    agg = {}
    for slide_id, v in bank.items():
        pid = slide_to_patient(slide_id)
        if pid not in agg:
            agg[pid] = {"feats": [v["feats"]], "slides": [slide_id]}
        else:
            agg[pid]["feats"].append(v["feats"])
            agg[pid]["slides"].append(slide_id)
    for pid in list(agg.keys()):
        agg[pid]["feats"] = np.concatenate(agg[pid]["feats"], axis=0)
        agg[pid]["num_tiles"] = int(agg[pid]["feats"].shape[0])
        agg[pid]["num_slides"] = len(agg[pid]["slides"])
    return agg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def predict_risk(mil, risk_head, bank_patient, ids, device):
    mil.eval()
    risk_head.eval()
    risks = []
    with torch.no_grad():
        for pid in ids:
            f = torch.tensor(bank_patient[pid]["feats"], dtype=torch.float32, device=device)
            z, _ = mil(f)
            r = risk_head(z).item()
            risks.append(r)
    return np.array(risks, dtype=np.float32)


def auc_at_t(times, events, risks, t):
    _, _, auc = time_dependent_roc(times, events, risks, t)
    return float(auc)


def train_fold(cfg, train_ids, bank_patient, df, in_dim, hidden, dropout, device, seed):
    set_seed(seed)
    bag_size = None
    lr = 1.0e-4
    weight_decay = 1.0e-4
    stab_w = 5.0e-4
    cox_l2 = 1.0e-4

    mil = build_mil(cfg["mil"].get("arch", "gated_attn"), in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    risk_head = nn.Linear(hidden, 1).to(device)
    opt = torch.optim.AdamW(list(mil.parameters()) + list(risk_head.parameters()), lr=lr, weight_decay=weight_decay)
    ds = BagDS(train_ids, bank_patient, df, bag_size=bag_size, strategy=cfg["mil"]["bag_strategy"])
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    best_mil = None
    best_head = None
    best_loss = 1e9
    bad = 0
    for ep in range(cfg["mil"]["epochs"]):
        mil.train()
        risk_head.train()
        opt.zero_grad()
        zs = []
        ts = []
        es = []
        stab_terms = []
        for _, bag, t, e in dl:
            bag = bag.squeeze(0).to(device)
            z, _ = mil(bag)
            zs.append(z)
            ts.append(float(t.item()))
            es.append(float(e.item()))
            stab_terms.append((z ** 2).mean())
        Z = torch.stack(zs, dim=0)
        risk = risk_head(Z).squeeze(-1)
        times_t = torch.tensor(ts, device=device, dtype=torch.float32)
        events_t = torch.tensor(es, device=device, dtype=torch.float32)
        loss_cox = cox_ph_loss(risk, times_t, events_t)
        loss_stab = torch.stack(stab_terms).mean()
        l2_head = risk_head.weight.pow(2).sum() + risk_head.bias.pow(2).sum()
        loss = loss_cox + stab_w * loss_stab + cox_l2 * l2_head
        loss.backward()
        opt.step()
        m = float(loss.item())
        if m < best_loss - 1e-4:
            best_loss = m
            best_mil = {k: v.detach().cpu() for k, v in mil.state_dict().items()}
            best_head = {k: v.detach().cpu() for k, v in risk_head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["mil"]["early_stop"]:
                break

    if best_mil is None or best_head is None:
        best_mil = {k: v.detach().cpu() for k, v in mil.state_dict().items()}
        best_head = {k: v.detach().cpu() for k, v in risk_head.state_dict().items()}

    mil.load_state_dict(best_mil)
    risk_head.load_state_dict(best_head)
    mil.eval()
    risk_head.eval()
    return mil, risk_head, best_mil, best_head


def run_cv(cfg, train_ids, val_ids, ext_ids, bank_patient, df, in_dim, hidden, dropout, device, base_seed, save_dir=None):
    train_ids = np.array([str(x) for x in train_ids], dtype=object)
    y = df.set_index("ID").loc[train_ids, "event"].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)

    oof_risk = np.zeros(len(train_ids), dtype=np.float32)
    val_preds = []
    ext_preds = []
    fold_states = []

    for fold_idx, (tr_idx, oof_idx) in enumerate(skf.split(train_ids, y), start=1):
        fold_train_ids = train_ids[tr_idx].tolist()
        fold_oof_ids = train_ids[oof_idx].tolist()
        mil, risk_head, best_mil, best_head = train_fold(
            cfg=cfg,
            train_ids=fold_train_ids,
            bank_patient=bank_patient,
            df=df,
            in_dim=in_dim,
            hidden=hidden,
            dropout=dropout,
            device=device,
            seed=base_seed + fold_idx,
        )
        oof_risk[oof_idx] = predict_risk(mil, risk_head, bank_patient, fold_oof_ids, device)
        if len(val_ids) > 0:
            val_preds.append(predict_risk(mil, risk_head, bank_patient, val_ids, device))
        if len(ext_ids) > 0:
            ext_preds.append(predict_risk(mil, risk_head, bank_patient, ext_ids, device))
        fold_states.append((best_mil, best_head))
        if save_dir is not None:
            torch.save(best_mil, os.path.join(save_dir, f"fold{fold_idx}_mil.pt"))
            torch.save(best_head, os.path.join(save_dir, f"fold{fold_idx}_head.pt"))

    val_mean = np.mean(np.stack(val_preds, axis=0), axis=0) if val_preds else np.array([], dtype=np.float32)
    ext_mean = np.mean(np.stack(ext_preds, axis=0), axis=0) if ext_preds else np.array([], dtype=np.float32)
    return oof_risk, val_mean, ext_mean, fold_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    wd = cfg["run"]["workdir"]
    ensure_dir(wd)
    ensure_dir(os.path.join(wd, "models"))
    log_dir = os.path.join(wd, "logs")
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "train_mil_cox.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    set_seed(int(cfg["run"]["seed"]))
    bank = load_bank(wd)
    bags_dir = os.path.join(wd, "patient_bags")
    if not os.path.isdir(bags_dir):
        logging.info(f"[BAGS] patient_bags not found at {bags_dir}, generating now...")
        bank_patient_tmp = aggregate_by_patient(bank)
        ensure_dir(bags_dir)
        for pid, v in bank_patient_tmp.items():
            out_npz = os.path.join(bags_dir, f"{pid}.npz")
            np.savez_compressed(out_npz, feats=v["feats"])
        logging.info(f"[BAGS] generated patient_bags at {bags_dir}")

    df = pd.read_excel(cfg["data"]["clinic_xlsx"])
    id_col = cfg["data"]["id_col"]
    status_col = cfg["data"]["status_col"]
    time_col = cfg["data"]["time_col"]
    center_col = cfg["data"]["center_col"]
    if status_col in df.columns and time_col in df.columns and center_col in df.columns and id_col in df.columns:
        df = df.rename(columns={status_col: "event", time_col: "time", center_col: "center", id_col: "ID"})
    else:
        s_i = int(cfg["data"].get("status_idx", 0))
        t_i = int(cfg["data"].get("time_idx", 1))
        c_i = int(cfg["data"].get("center_idx", 2))
        id_i = int(cfg["data"].get("id_idx", 3))
        df = df.iloc[:, [s_i, t_i, c_i, id_i]].copy()
        df.columns = ["event", "time", "center", "ID"]

    bank_patient = {}
    for f in sorted(glob.glob(os.path.join(bags_dir, "*.npz"))):
        pid = os.path.splitext(os.path.basename(f))[0]
        data = np.load(f)
        bank_patient[pid] = {"feats": data["feats"].astype("float32")}

    keys = sorted(bank_patient.keys())
    df = df[df["ID"].astype(str).isin(keys)].copy()
    keep_ids = set(df["ID"].astype(str).values.tolist())
    bank_patient = {k: v for k, v in bank_patient.items() if k in keep_ids}
    df["ID"] = df["ID"].astype(str)
    df["event"] = df["event"].astype(int)
    df["time"] = df["time"].astype(float)

    center = df["center"].astype(str).str.strip().str.upper()
    z1 = center == "Z1"
    id_upper = df["ID"].astype(str).str.upper()
    z1_prefix = id_upper.str.startswith(("YP", "ZP"))
    df_pool = df[z1 & ~z1_prefix].copy()
    df_ext = df[(z1 & z1_prefix) | center.isin(["S3", "Z3", "SD"])].copy()

    tr_ids, va_ids = train_test_split(
        df_pool["ID"].values,
        test_size=cfg["split"]["val_ratio"],
        random_state=cfg["run"]["seed"],
        stratify=df_pool["event"].values if cfg["split"]["stratify_by_event"] else None,
    )
    te_ids = df_ext["ID"].values
    tr_ids = [str(x) for x in tr_ids]
    va_ids = [str(x) for x in va_ids]
    te_ids = [str(x) for x in te_ids]
    logging.info(f"[SPLIT] train={len(tr_ids)} val={len(va_ids)} external={len(te_ids)}")

    HIDDEN_GRID = [64, 128, 256]
    DROPOUT_GRID = [0.2, 0.3, 0.4]
    TIME_ROC_TS = [7, 14, 28, 60]
    BAG_SIZE = None
    LR = 1.0e-4
    WEIGHT_DECAY = 1.0e-4
    STAB_W = 5.0e-4
    COX_L2 = 1.0e-4

    in_dim = list(bank_patient.values())[0]["feats"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_cindex = -1e9
    best_hidden = None
    best_dropout = None
    for hidden in HIDDEN_GRID:
        for dropout in DROPOUT_GRID:
            oof_train, _, _, _ = run_cv(
                cfg=cfg,
                train_ids=tr_ids,
                val_ids=[],
                ext_ids=[],
                bank_patient=bank_patient,
                df=df,
                in_dim=in_dim,
                hidden=hidden,
                dropout=dropout,
                device=device,
                base_seed=int(cfg["run"]["seed"]),
                save_dir=None,
            )
            train_df = df.set_index("ID").loc[tr_ids]
            cindex = safe_cindex(train_df["time"].values, -oof_train, train_df["event"].values)
            auc7 = auc_at_t(train_df["time"].values, train_df["event"].values, oof_train, 7)
            auc14 = auc_at_t(train_df["time"].values, train_df["event"].values, oof_train, 14)
            auc28 = auc_at_t(train_df["time"].values, train_df["event"].values, oof_train, 28)
            auc60 = auc_at_t(train_df["time"].values, train_df["event"].values, oof_train, 60)
            logging.info(
                f"[HPARAM] hidden={hidden} dropout={dropout:.1f} "
                f"C-index={cindex:.4f} "
                f"AUC@7d={auc7:.4f} AUC@14d={auc14:.4f} AUC@28d={auc28:.4f} AUC@60d={auc60:.4f}"
            )
            if np.isnan(cindex):
                continue
            if cindex > best_cindex:
                best_cindex = cindex
                best_hidden = hidden
                best_dropout = dropout

    if best_hidden is None or best_dropout is None:
        raise RuntimeError("Hyperparameter search failed: no valid C-index was produced.")

    logging.info(f"[BEST] hidden={best_hidden} dropout={best_dropout:.1f} C-index={best_cindex:.4f}")
    with open(os.path.join(wd, "models", "best_hparams.json"), "w") as f:
        json.dump(
            {
                "hidden": best_hidden,
                "dropout": best_dropout,
                "best_cv_cindex": float(best_cindex),
                "bag_size": BAG_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "stability_weight": STAB_W,
                "cox_l2": COX_L2,
            },
            f,
            indent=2,
        )

    r_tr, r_va, r_te, _ = run_cv(
        cfg=cfg,
        train_ids=tr_ids,
        val_ids=va_ids,
        ext_ids=te_ids,
        bank_patient=bank_patient,
        df=df,
        in_dim=in_dim,
        hidden=best_hidden,
        dropout=best_dropout,
        device=device,
        base_seed=int(cfg["run"]["seed"]),
        save_dir=os.path.join(wd, "models"),
    )

    train_df = df.set_index("ID").loc[tr_ids]
    val_df = df.set_index("ID").loc[va_ids]
    ext_df = df.set_index("ID").loc[te_ids] if len(te_ids) > 0 else pd.DataFrame(columns=["time", "event"])

    Ttr, Etr = train_df["time"].values, train_df["event"].values
    Tva, Eva = val_df["time"].values, val_df["event"].values
    Tte = ext_df["time"].values if len(te_ids) > 0 else np.array([])
    Ete = ext_df["event"].values if len(te_ids) > 0 else np.array([])

    out_scores = os.path.join(wd, "scores.csv")
    rows = []
    for _id, t, e, r in zip(tr_ids, Ttr, Etr, r_tr):
        rows.append({"ID": _id, "time": float(t), "event": int(e), "risk": float(r), "split": "train"})
    for _id, t, e, r in zip(va_ids, Tva, Eva, r_va):
        rows.append({"ID": _id, "time": float(t), "event": int(e), "risk": float(r), "split": "val"})
    for _id, t, e, r in zip(te_ids, Tte, Ete, r_te):
        rows.append({"ID": _id, "time": float(t), "event": int(e), "risk": float(r), "split": "external"})
    pd.DataFrame(rows).to_csv(out_scores, index=False)
    logging.info(f"[SAVE] scores -> {out_scores}")

    cutoff = float(np.median(r_tr))
    with open(os.path.join(wd, "models", "cutoff.json"), "w") as f:
        json.dump({"cutoff": cutoff}, f, indent=2)
    logging.info(f"[CUT] cutoff={cutoff:.4f} (train median)")

    km_plot_by_cutoff(Ttr, Etr, r_tr, cutoff, "KM on TRAIN (median cutoff)", os.path.join(wd, "km_train.png"))
    km_plot_by_cutoff(Tva, Eva, r_va, cutoff, "KM on VAL (median cutoff)", os.path.join(wd, "km_val.png"))
    if len(te_ids) > 0:
        km_plot_by_cutoff(Tte, Ete, r_te, cutoff, "KM on EXTERNAL (median cutoff)", os.path.join(wd, "km_external.png"))

    splits = [
        {"name": "train", "times": Ttr, "events": Etr, "risks": r_tr},
        {"name": "val", "times": Tva, "events": Eva, "risks": r_va},
    ]
    if len(te_ids) > 0:
        splits.append({"name": "external", "times": Tte, "events": Ete, "risks": r_te})
    for t in TIME_ROC_TS:
        roc_path = os.path.join(wd, f"timeROC_{int(t)}d.png")
        time_roc_plot(splits, t, roc_path, title="Time-dependent ROC")
        logging.info(f"[SAVE] timeROC -> {roc_path}")

    def metrics_block(times, events, risks):
        return {
            "cindex": safe_cindex(times, -risks, events),
            "auc7": auc_at_t(times, events, risks, 7),
            "auc14": auc_at_t(times, events, risks, 14),
            "auc28": auc_at_t(times, events, risks, 28),
            "auc60": auc_at_t(times, events, risks, 60),
        }

    m_tr = metrics_block(Ttr, Etr, r_tr)
    m_va = metrics_block(Tva, Eva, r_va)
    m_te = metrics_block(Tte, Ete, r_te) if len(te_ids) > 0 else {"cindex": np.nan, "auc7": np.nan, "auc14": np.nan, "auc28": np.nan, "auc60": np.nan}

    report_path = os.path.join(wd, "report.txt")
    with open(report_path, "w") as f:
        f.write(
            f"Best hyperparams: hidden={best_hidden}, dropout={best_dropout}\n"
            f"Best CV C-index: {best_cindex:.4f}\n"
            f"Other hyperparams: bag_size={BAG_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}, stability={STAB_W}, cox_l2={COX_L2}\n"
            f"timeROC_ts={TIME_ROC_TS} (days)\n\n"
            f"--- Training (5-fold CV out-of-fold, n={len(tr_ids)}) ---\n"
            f"C-index: {m_tr['cindex']:.4f}\n"
            f"AUC@7d: {m_tr['auc7']:.4f}\n"
            f"AUC@14d: {m_tr['auc14']:.4f}\n"
            f"AUC@28d: {m_tr['auc28']:.4f}\n"
            f"AUC@60d: {m_tr['auc60']:.4f}\n"
            f"\n"
            f"--- Internal Validation (n={len(va_ids)}) ---\n"
            f"C-index: {m_va['cindex']:.4f}\n"
            f"AUC@7d: {m_va['auc7']:.4f}\n"
            f"AUC@14d: {m_va['auc14']:.4f}\n"
            f"AUC@28d: {m_va['auc28']:.4f}\n"
            f"AUC@60d: {m_va['auc60']:.4f}\n"
            f"\n"
            f"--- External Validation (n={len(te_ids)}) ---\n"
            f"C-index: {m_te['cindex']:.4f}\n"
            f"AUC@7d: {m_te['auc7']:.4f}\n"
            f"AUC@14d: {m_te['auc14']:.4f}\n"
            f"AUC@28d: {m_te['auc28']:.4f}\n"
            f"AUC@60d: {m_te['auc60']:.4f}\n"
            f"\n"
        )
    logging.info(f"[SAVE] report -> {report_path}")


if __name__ == "__main__":
    main()
