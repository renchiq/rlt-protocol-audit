"""Основной эксперимент: модели x протоколы разбиения x повторы (seeds).

Для каждой пары (модель, протокол) и каждого seed выполняется полная k-fold кросс-валидация по
заранее сохранённым разбиениям splits/<protocol>_seed<seed>.json. Сохраняются out-of-fold
вероятности для каждого клипа (results/predictions/) и метрики по каждому повтору
(results/metrics_raw.csv).

Запуск:  python src/run_experiment.py --config configs/experiment.yaml [--models a,b] [--protocols clip,speaker]
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import models as dl
from splits import inner_splitter

ROOT = Path(__file__).resolve().parents[1]
DEEP = {"transformer", "transformer_shuffled", "gru", "framepool", "transformer_es", "gru_es", "framepool_es"}


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(y, p), acc=accuracy_score(y, yhat), bacc=balanced_accuracy_score(y, yhat),
                f1_macro=f1_score(y, yhat, average="macro"))


def features_for(model_name, meta, cfg):
    clips = meta["clip"].tolist()
    fs = cfg["feature_set"]
    if model_name in ("stat_logreg", "stat_rf"):
        return data.stat_matrix(clips, fs, include_delta=True)
    if model_name == "stat_nodelta_logreg":
        return data.stat_matrix(clips, fs, include_delta=False)
    if model_name == "gesture_logreg":
        return data.gesture_matrix(clips)[0]
    if model_name == "metadata_rf":
        return data.metadata_matrix(meta)
    if model_name == "identity_1nn":
        return data.sface_matrix(clips)
    if model_name == "prior":
        return np.zeros((len(clips), 1), np.float32)
    if model_name == "gender":
        return (meta.gender == "F").to_numpy(np.float32)[:, None]
    if model_name in DEEP:
        return data.load_sequences(clips, fs)[0]
    raise ValueError(model_name)


def fit_predict_tabular(name, Xtr, ytr, gtr, Xte, protocol, cfg, seed):
    if name in ("stat_logreg", "stat_nodelta_logreg", "gesture_logreg"):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        inner = inner_splitter(protocol, cfg["inner_splits"], seed)
        gs = GridSearchCV(pipe, {"logisticregression__C": cfg["logreg_C_grid"]}, scoring="roc_auc", cv=inner)
        gs.fit(Xtr, ytr, groups=None if protocol == "clip" else gtr)
        return gs.predict_proba(Xte)[:, 1], dict(C=gs.best_params_["logisticregression__C"])
    if name in ("stat_rf", "metadata_rf"):
        rf = RandomForestClassifier(n_estimators=cfg["rf"]["n_estimators"],
                                    min_samples_leaf=cfg["rf"]["min_samples_leaf"], random_state=seed, n_jobs=1)
        rf.fit(Xtr, ytr)
        return rf.predict_proba(Xte)[:, 1], {}
    if name == "prior":
        # Доля обманных клипов в обучающей части фолда — диагностика сдвига априорной вероятности.
        return np.full(len(Xte), ytr.mean()), {}
    if name == "gender":
        # Доля обманных клипов среди клипов того же пола в обучающей части (демографический «короткий путь»).
        f = Xtr[:, 0] == 1
        rate = {1.0: ytr[f].mean() if f.any() else ytr.mean(), 0.0: ytr[~f].mean() if (~f).any() else ytr.mean()}
        return np.array([rate[float(v)] for v in Xte[:, 0]]), {}
    if name == "identity_1nn":
        # Метка обучающего клипа с ближайшим дескриптором лица (SFace) (косинусная близость): чистый «идентификационный» путь.
        a = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
        b = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
        nn_idx = (b @ a.T).argmax(1)
        return ytr[nn_idx].astype(float), {}
    raise ValueError(name)


def run_fold(name, X, y, groups, fold, k, protocol, cfg, seed):
    torch.set_num_threads(1)
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    t0 = time.time()
    if name in DEEP:
        inner = None
        if name.endswith("_es"):
            # внутренняя валидация для ранней остановки: первый фолд внутреннего разбиения того же типа
            sp = inner_splitter(protocol, cfg["inner_splits"], seed)
            inner = next(sp.split(tr, y[tr], None if protocol == "clip" else groups[tr]))
        p, last_loss, es_info = dl.train_and_predict(name, [X[i] for i in tr], y[tr], [X[i] for i in te],
                                                     cfg["deep"], seed * 1000 + k, inner=inner)
        info = dict(final_train_loss=last_loss, **es_info)
    else:
        p, info = fit_predict_tabular(name, X[tr], y[tr], groups[tr], X[te], protocol, cfg, seed)
    return te, p, dict(fold=int(k), n_test=len(te), seconds=round(time.time() - t0, 2), **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models")
    ap.add_argument("--protocols")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model_names = args.models.split(",") if args.models else cfg["models"]
    protocols = args.protocols.split(",") if args.protocols else cfg["protocols"]
    out = ROOT / cfg["results_dir"]
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs/run_experiment.log"), logging.StreamHandler()])
    meta = data.load_meta()
    y = meta.label.to_numpy()
    raw_path = out / "metrics_raw.csv"
    for name in model_names:
        X = features_for(name, meta, cfg)
        for protocol in protocols:
            groups = meta[{"clip": "clip", "speaker": "speaker_id", "case": "case_id"}[protocol]].to_numpy()
            rows, preds = [], []
            t0 = time.time()
            for seed in cfg["seeds"]:
                sp = json.load(open(ROOT / f"splits/{protocol}_seed{seed}.json"))
                assert sp["clips"] == meta["clip"].tolist()
                fold = np.array(sp["fold"])
                res = Parallel(n_jobs=cfg["n_jobs"])(
                    delayed(run_fold)(name, X, y, groups, fold, k, protocol, cfg, seed)
                    for k in range(cfg["n_splits"]))
                p = np.zeros(len(y))
                best_epoch = np.zeros(len(y), int)
                for te, pk, info in res:
                    p[te] = pk
                    best_epoch[te] = info.get("best_epoch", 0)
                    logging.info(json.dumps(dict(model=name, protocol=protocol, seed=seed, **info)))
                m = metrics(y, p)
                rows.append(dict(model=name, protocol=protocol, seed=seed, **m))
                pr = pd.DataFrame(dict(clip=meta["clip"], label=y, seed=seed, fold=fold, prob=p))
                if name.endswith("_es"):
                    pr["best_epoch"] = best_epoch  # число эпох, выбранное ранней остановкой для фолда клипа
                preds.append(pr)
                logging.info(f"{name} {protocol} seed={seed} " + " ".join(f"{a}={b:.3f}" for a, b in m.items()))
            pd.concat(preds).to_csv(out / f"predictions/{name}__{protocol}.csv", index=False)
            df = pd.DataFrame(rows)
            if raw_path.exists():
                old = pd.read_csv(raw_path)
                old = old[~((old.model == name) & (old.protocol == protocol))]
                df = pd.concat([old, df])
            df.to_csv(raw_path, index=False)
            logging.info(f"DONE {name} {protocol}: AUC={np.mean([r['auc'] for r in rows]):.3f} "
                         f"acc={np.mean([r['acc'] for r in rows]):.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
