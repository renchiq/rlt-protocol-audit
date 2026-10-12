"""Протоколы разбиения для кросс-валидации.

clip    — стратифицированный k-fold по клипам (стандартная практика многих работ; допускает
          попадание одного говорящего и в обучение, и в тест — subject leakage);
speaker — StratifiedGroupKFold по идентификатору говорящего (говорящий целиком в одной части);
case    — StratifiedGroupKFold по судебному делу/источнику (все клипы одного дела в одной части;
          строже speaker: исключает утечку через общую обстановку записи и тему).

Запуск:  python src/splits.py --config configs/experiment.yaml   -> splits/<protocol>_seed<k>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]


def make_folds(meta, protocol, n_splits, seed):
    y = meta.label.to_numpy()
    idx = np.arange(len(meta))
    if protocol == "clip":
        it = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(idx, y)
    else:
        groups = meta[{"speaker": "speaker_id", "case": "case_id"}[protocol]].to_numpy()
        it = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(idx, y, groups)
    fold = np.full(len(meta), -1)
    for k, (_, te) in enumerate(it):
        fold[te] = k
    assert (fold >= 0).all()
    return fold


def inner_splitter(protocol, n_splits, seed):
    if protocol == "clip":
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)


def check_leakage(meta, fold, protocol):
    """Число групп (говорящих/дел), попадающих в несколько фолдов."""
    rep = {}
    for g in ("speaker_id", "case_id"):
        rep[g] = int((meta.assign(f=fold).groupby(g).f.nunique() > 1).sum())
    return rep


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    meta = pd.read_csv(ROOT / "metadata/clips.csv")
    out = ROOT / "splits"
    out.mkdir(exist_ok=True)
    summary = []
    for protocol in cfg["protocols"]:
        for seed in cfg["seeds"]:
            fold = make_folds(meta, protocol, cfg["n_splits"], seed)
            leak = check_leakage(meta, fold, protocol)
            json.dump(dict(protocol=protocol, seed=seed, clips=meta["clip"].tolist(), fold=fold.tolist(),
                           groups_in_multiple_folds=leak),
                      open(out / f"{protocol}_seed{seed}.json", "w"))
            summary.append(dict(protocol=protocol, seed=seed, **{f"leak_{k}": v for k, v in leak.items()},
                                min_test=int(np.bincount(fold).min()), max_test=int(np.bincount(fold).max())))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summary).groupby("protocol").mean(numeric_only=True))


if __name__ == "__main__":
    main()
