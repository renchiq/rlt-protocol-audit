"""Графики – строятся только из results/aggregate/*.csv и results/audit/*.

Запуск:  python src/figures.py --config configs/aggregate.yaml   -> results/figures/*.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTO_RU = {"clip": "по клипам", "speaker": "по говорящим", "case": "по делам"}
MODEL_RU = {
    "prior": "Априорная доля классов",
    "gender": "Только пол говорящего",
    "identity_1nn": "1-NN по дескриптору лица",
    "metadata_rf": "RF: техн. параметры видео",
    "gesture_logreg": "LR: ручная разметка жестов",
    "stat_nodelta_logreg": "LR: статистики (без Δ)",
    "stat_logreg": "LR: статистики + |Δ|",
    "stat_rf": "RF: статистики + |Δ|",
    "framepool": "MLP + усреднение по времени",
    "gru": "BiGRU",
    "transformer": "Transformer",
    "transformer_shuffled": "Transformer (кадры перемешаны)",
    "framepool_es": "MLP, ранняя остановка",
    "gru_es": "BiGRU, ранняя остановка",
    "transformer_es": "Transformer, ранняя остановка",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    res = ROOT / cfg["results_dir"]
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(res / "aggregate/main_table.csv")
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

    for metric, xlabel, chance in (("auc", "ROC AUC", 0.5), ("auc_fold", "ROC AUC, усреднённый по блокам", 0.5),
                                   ("bacc", "Сбалансированная точность", 0.5)):
        models = [m for m in cfg["models"] if m in set(tab.model)]
        fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(models) + 1.0))
        offs = {"clip": -0.25, "speaker": 0.0, "case": 0.25}
        colors = {"clip": "#b2182b", "speaker": "#2166ac", "case": "#4d9221"}
        markers = {"clip": "o", "speaker": "s", "case": "D"}
        for pr in cfg["protocols"]:
            for i, m in enumerate(models):
                r = tab[(tab.model == m) & (tab.protocol == pr)]
                if not len(r):
                    continue
                r = r.iloc[0]
                yv = len(models) - 1 - i + offs[pr]
                ax.errorbar(r[f"{metric}_mean"], yv,
                            xerr=[[r[f"{metric}_mean"] - r[f"{metric}_ci_lo"]], [r[f"{metric}_ci_hi"] - r[f"{metric}_mean"]]],
                            fmt=markers[pr], color=colors[pr], ms=4, capsize=2, lw=1,
                            label=PROTO_RU[pr] if i == 0 else None)
        ax.axvline(chance, color="grey", ls="--", lw=0.8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_RU.get(m, m) for m in models][::-1])
        ax.set_xlabel(f"{xlabel}\nсреднее по 5 повторам; 95 % ДИ – кластерный бутстрэп")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False, title="Разбиение:", title_fontsize=8)
        ax.set_xlim(0.2, 1.0)
        fig.tight_layout()
        fig.savefig(out / f"fig_{metric}_by_protocol.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    comp = pd.read_csv(res / "aggregate/paired_comparisons.csv")
    c = comp[(comp.kind == "protocol") & (comp.metric == "auc_fold") & (comp.b.str.endswith("@speaker"))
             & (comp.a.str.endswith("@clip"))].copy()
    if len(c):
        c["model"] = c.a.str.split("@").str[0]
        c = c.set_index("model").loc[[m for m in cfg["models"] if m in set(c.model) and m != "prior"]]
        fig, ax = plt.subplots(figsize=(6.0, 0.4 * len(c) + 1.0))
        yv = np.arange(len(c))[::-1]
        ax.errorbar(c.delta, yv, xerr=[c.delta - c.ci_lo, c.ci_hi - c.delta], fmt="o", color="#b2182b", capsize=2)
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        ax.set_yticks(yv)
        ax.set_yticklabels([MODEL_RU.get(m, m) for m in c.index])
        ax.set_xlabel("ΔAUC (усреднённый по блокам) = по клипам − по говорящим\n95 % ДИ, кластерный бутстрэп по говорящим")
        fig.tight_layout()
        fig.savefig(out / "fig_leakage_inflation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    # Структура датасета: клипы на говорящего с разбивкой по меткам
    g = pd.read_csv(res / "audit/clips_per_speaker.csv").sort_values("count", ascending=False)
    g["lie"] = (g["count"] * g["mean"]).round().astype(int)
    g["truth"] = g["count"] - g["lie"]
    # размер 1:1 со вставкой в статью (14 см), шрифт ≥ 10 пт в печати
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(5.5, 2.1))
    x = np.arange(len(g))
    ax.bar(x, g["lie"], color="#b2182b", label="обман")
    ax.bar(x, g["truth"], bottom=g["lie"], color="#2166ac", label="правда")
    ax.set_xlabel("говорящие (по убыванию числа клипов)")
    ax.set_ylabel("число клипов")
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_clips_per_speaker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figures:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
