"""Агрегация результатов: средние по повторам, кластерный бутстрэп по говорящим (95 % ДИ),
парные сравнения протоколов и моделей. Все числа берутся только из results/predictions/*.csv
и сохранённых разбиений splits/*.json.

Метрика auc_fold — ROC AUC, вычисленный отдельно внутри каждого тестового фолда (фолды с обоими
классами) и усреднённый. В отличие от объединённого (pooled) AUC по всем out-of-fold предсказаниям,
она не зависит от различий калибровки между фолдами, в частности от сдвига априорной доли классов
в обучающих частях при групповой кросс-валидации.

Запуск:  python src/aggregate.py --config configs/aggregate.yaml
Результат: results/aggregate/*.csv, *.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def load_preds(res, model, protocol):
    p = res / f"predictions/{model}__{protocol}.csv"
    return pd.read_csv(p) if p.exists() else None


def metric_seeds(name, y, M):
    """Метрика для каждого повтора сразу: y [n], M [n_seeds, n] -> [n_seeds].
    AUC — через ранги (статистика Манна–Уитни, ties усредняются), совпадает с roc_auc_score."""
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if name == "auc":
        if n1 == 0 or n0 == 0:
            return np.full(M.shape[0], np.nan)
        r = rankdata(M, axis=1)
        return (r[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    yhat = M >= 0.5
    if name == "acc":
        return (yhat == pos[None, :]).mean(1)
    if name == "bacc":
        tpr = yhat[:, pos].mean(1) if n1 else np.full(M.shape[0], np.nan)
        tnr = (~yhat[:, ~pos]).mean(1) if n0 else np.full(M.shape[0], np.nan)
        return (tpr + tnr) / 2
    raise ValueError(name)


def seed_matrix(df, clips):
    """[n_seeds, n_clips] вероятностей в порядке clips."""
    piv = df.pivot(index="seed", columns="clip", values="prob")[clips]
    return piv.to_numpy()


def boot_indices(meta, B, rng):
    """Кластерный бутстрэп: выборка говорящих с возвращением, все их клипы входят в выборку."""
    spk = meta.speaker_id.to_numpy()
    uniq = np.unique(spk)
    members = {s: np.where(spk == s)[0] for s in uniq}
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        yield np.concatenate([members[s] for s in pick])


def auc_fold_seeds(y, M, F):
    """Средний по фолдам AUC для каждого повтора: y [n], M, F [n_seeds, n] (F — номер фолда)."""
    out = []
    for s in range(M.shape[0]):
        vals = []
        for f in np.unique(F[s]):
            m = F[s] == f
            yy = y[m]
            n1, n0 = (yy == 1).sum(), (yy == 0).sum()
            if n1 and n0:
                r = rankdata(M[s, m])
                vals.append((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def per_seed(name, y, P, idx):
    if name == "auc_fold":
        return auc_fold_seeds(y[idx], P["raw"][:, idx], P["fold"][:, idx])
    return metric_seeds(name, y[idx], P["raw"][:, idx])


def mean_over_seeds(name, y, P, idx=None):
    """P — словарь {"raw": [n_seeds, n_clips] вероятностей, "fold": [n_seeds, n_clips] номеров фолдов}."""
    if idx is None:
        idx = np.arange(len(y))
    return float(np.nanmean(per_seed(name, y, P, idx)))


def fold_matrix(protocol, seeds, clips):
    out = []
    for seed in seeds:
        sp = json.load(open(ROOT / f"splits/{protocol}_seed{seed}.json"))
        assert sp["clips"] == clips
        out.append(np.array(sp["fold"]))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    res = ROOT / cfg["results_dir"]
    out = res / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "metadata/clips.csv")
    clips = meta["clip"].tolist()
    y = meta.label.to_numpy()
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    boots = list(boot_indices(meta, cfg["n_bootstrap"], rng))

    P = {}
    folds = {pr: fold_matrix(pr, cfg["seeds"], clips) for pr in cfg["protocols"]}
    for m in cfg["models"]:
        for pr in cfg["protocols"]:
            df = load_preds(res, m, pr)
            if df is not None:
                assert sorted(df.seed.unique()) == cfg["seeds"]
                fm = df.pivot(index="seed", columns="clip", values="fold")[clips].to_numpy()
                assert (fm == folds[pr]).all()
                P[(m, pr)] = {"raw": seed_matrix(df, clips), "fold": fm}

    # 1) Основная таблица: среднее ± std по повторам + кластерный бутстрэп-ДИ
    rows = []
    for (m, pr), M in P.items():
        row = dict(model=m, protocol=pr, n_seeds=M["raw"].shape[0])
        for name in cfg["metrics"]:
            bs = np.array([mean_over_seeds(name, y, M, b) for b in boots])
            ps = per_seed(name, y, M, np.arange(len(y)))
            row[f"{name}_mean"] = float(np.mean(ps))
            row[f"{name}_std_seeds"] = float(np.std(ps, ddof=1))
            row[f"{name}_ci_lo"], row[f"{name}_ci_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(row)
    main_tab = pd.DataFrame(rows)
    main_tab.to_csv(out / "main_table.csv", index=False)

    # 2) Парные разности (один и тот же бутстрэп-набор говорящих для обоих членов разности)
    def paired(a, b, name):
        Ma, Mb = P[a], P[b]
        d0 = mean_over_seeds(name, y, Ma) - mean_over_seeds(name, y, Mb)
        ds = np.array([mean_over_seeds(name, y, Ma, bi) - mean_over_seeds(name, y, Mb, bi) for bi in boots])
        lo, hi = np.nanpercentile(ds, [2.5, 97.5])
        p_two = 2 * min(np.nanmean(ds <= 0), np.nanmean(ds >= 0))
        return dict(a=f"{a[0]}@{a[1]}", b=f"{b[0]}@{b[1]}", metric=name, delta=d0, ci_lo=lo, ci_hi=hi,
                    p_boot=min(p_two, 1.0))

    comp = []
    for m in cfg["models"]:
        for pa, pb in cfg["protocol_pairs"]:
            if (m, pa) in P and (m, pb) in P:
                for name in cfg["metrics"]:
                    comp.append(dict(kind="protocol", **paired((m, pa), (m, pb), name)))
    for ma, mb in cfg["model_pairs"]:
        for pr in cfg["protocols"]:
            if (ma, pr) in P and (mb, pr) in P:
                for name in cfg["metrics"]:
                    comp.append(dict(kind="model", **paired((ma, pr), (mb, pr), name)))
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "paired_comparisons.csv", index=False)

    # 2б) Фолды, по которым считается auc_fold: тестовые фолды с клипами обоих классов
    fs = []
    for pr in cfg["protocols"]:
        for si, seed in enumerate(cfg["seeds"]):
            F = folds[pr][si]
            both = [f for f in np.unique(F) if len(np.unique(y[F == f])) == 2]
            fs.append(dict(protocol=pr, seed=seed, n_folds=len(np.unique(F)), folds_both_classes=len(both),
                           clips_in_used_folds=int(np.isin(F, both).sum())))
    fs = pd.DataFrame(fs)
    fs.to_csv(out / "fold_summary.csv", index=False)
    print(fs.groupby("protocol")[["folds_both_classes", "clips_in_used_folds"]].agg(["min", "mean", "max"]))

    # 2в) Ранняя остановка: число эпох, выбранное по внутренней валидации (по фолдам и повторам)
    es = []
    for m in cfg["models"]:
        for pr in cfg["protocols"]:
            df = load_preds(res, m, pr)
            if df is not None and "best_epoch" in df:
                e = df.groupby(["seed", "fold"]).best_epoch.first()
                es.append(dict(model=m, protocol=pr, n=len(e), median=float(e.median()), q1=float(e.quantile(0.25)),
                               q3=float(e.quantile(0.75)), share_le5=float((e <= 5).mean())))
    if es:
        pd.DataFrame(es).to_csv(out / "early_stopping.csv", index=False)
        print(pd.DataFrame(es))

    # 3) Markdown-версии таблиц
    def fmt(r, name):
        return f"{r[f'{name}_mean']:.3f} ± {r[f'{name}_std_seeds']:.3f} [{r[f'{name}_ci_lo']:.3f}; {r[f'{name}_ci_hi']:.3f}]"

    with open(out / "main_table.md", "w") as f:
        for name in cfg["metrics"]:
            f.write(f"\n### {name.upper()} (среднее ± std по {cfg['n_seeds_expected']} повторам "
                    f"[95 % ДИ, кластерный бутстрэп по говорящим, B={cfg['n_bootstrap']}])\n\n")
            f.write("| Модель | " + " | ".join(cfg["protocols"]) + " |\n|---|" + "---|" * len(cfg["protocols"]) + "\n")
            for m in cfg["models"]:
                cells = []
                for pr in cfg["protocols"]:
                    r = main_tab[(main_tab.model == m) & (main_tab.protocol == pr)]
                    cells.append(fmt(r.iloc[0], name) if len(r) else "—")
                f.write(f"| {m} | " + " | ".join(cells) + " |\n")
    with open(out / "paired_comparisons.md", "w") as f:
        f.write("| Тип | A | B | Метрика | Δ = A − B | 95 % ДИ | p (бутстрэп) |\n|---|---|---|---|---|---|---|\n")
        for _, r in comp.iterrows():
            f.write(f"| {r.kind} | {r.a} | {r.b} | {r.metric} | {r.delta:+.3f} | [{r.ci_lo:+.3f}; {r.ci_hi:+.3f}] | {r.p_boot:.3f} |\n")
    print(open(out / "main_table.md").read())


if __name__ == "__main__":
    main()
