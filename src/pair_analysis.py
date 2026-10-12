"""Декомпозиция AUC по типам пар (обман, правда) внутри тестовых фолдов.

AUC — это доля правильно упорядоченных пар (обманный клип, правдивый клип). Пары разделяются на:
  within  — оба клипа от одного говорящего (только 5 говорящих имеют клипы обоих классов);
  between — клипы разных говорящих.
Within-AUC показывает, различает ли модель обман и правду у одного человека; between-AUC —
насколько ранжирование определяется различиями между людьми. Учитываются только пары внутри
одного тестового фолда (не зависит от межфолдовой калибровки). Для каждого повтора доля
правильно упорядоченных пар (ничьи = 0,5) усредняется по повторам; интервал — кластерный
бутстрэп по говорящим (B из конфига).

Запуск:  python src/pair_analysis.py --config configs/aggregate.yaml  -> results/aggregate/pair_auc.csv|md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def pair_auc(y, spk, p, fold, idx):
    """Возвращает (within, between) долю правильно упорядоченных пар по набору индексов idx
    (индексы могут повторяться — бутстрэп; повтор говорящего трактуется как отдельный говорящий)."""
    out = {"within": [0.0, 0], "between": [0.0, 0]}
    for f in np.unique(fold[idx]):
        sel = idx[fold[idx] == f]
        pos, neg = sel[y[sel] == 1], sel[y[sel] == 0]
        if not len(pos) or not len(neg):
            continue
        cmp = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
        same = spk[pos][:, None] == spk[neg][None, :]
        out["within"][0] += cmp[same].sum()
        out["within"][1] += same.sum()
        out["between"][0] += cmp[~same].sum()
        out["between"][1] += (~same).sum()
    return {k: (v[0] / v[1] if v[1] else np.nan) for k, v in out.items()} | {
        f"n_{k}": v[1] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    res = ROOT / cfg["results_dir"]
    meta = pd.read_csv(ROOT / "metadata/clips.csv")
    clips = meta["clip"].tolist()
    y = meta.label.to_numpy()
    # Идентификатор «говорящий» при бутстрэпе: повторно выбранный говорящий получает новый id,
    # чтобы пары между его копиями не считались within.
    spk_codes = pd.factorize(meta.speaker_id)[0]
    members = {s: np.where(spk_codes == s)[0] for s in np.unique(spk_codes)}
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    boots = []
    for _ in range(cfg["n_bootstrap"]):
        pick = rng.choice(list(members), len(members), replace=True)
        idx = np.concatenate([members[s] for s in pick])
        sid = np.concatenate([np.full(len(members[s]), k) for k, s in enumerate(pick)])
        boots.append((idx, sid))
    rows = []
    for m in cfg["models"]:
        for pr in cfg["protocols"]:
            f = res / f"predictions/{m}__{pr}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            P = df.pivot(index="seed", columns="clip", values="prob")[clips].to_numpy()
            F = df.pivot(index="seed", columns="clip", values="fold")[clips].to_numpy()
            point = [pair_auc(y, spk_codes, P[s], F[s], np.arange(len(y))) for s in range(len(P))]
            row = dict(model=m, protocol=pr, n_within_pairs=point[0]["n_within"], n_between_pairs=point[0]["n_between"])
            bs = {"within": [], "between": []}
            for idx, sid in boots:
                ds = [pair_auc(y[idx], sid, P[s][idx], F[s][idx], np.arange(len(idx))) for s in range(len(P))]
                for k in bs:
                    bs[k].append(np.nanmean([d[k] for d in ds]))
            for k in ("within", "between"):
                row[k] = float(np.mean([d[k] for d in point]))
                row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
            rows.append(row)
            print(m, pr, f"within={row['within']:.3f} between={row['between']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(res / "aggregate/pair_auc.csv", index=False)
    with open(res / "aggregate/pair_auc.md", "w") as fh:
        fh.write("| Модель | Протокол | within-AUC [95 % ДИ] | between-AUC [95 % ДИ] | пар within / between |\n|---|---|---|---|---|\n")
        for _, r in out.iterrows():
            fh.write(f"| {r["model"]} | {r.protocol} | {r["within"]:.3f} [{r.within_ci_lo:.3f}; {r.within_ci_hi:.3f}] | "
                     f"{r["between"]:.3f} [{r.between_ci_lo:.3f}; {r.between_ci_hi:.3f}] | {r.n_within_pairs} / {r.n_between_pairs} |\n")


if __name__ == "__main__":
    main()
