"""Аудит структуры датасета и «коротких путей».

1) Описательная статистика: число говорящих/дел, доля говорящих с клипами обоих классов,
   распределение клипов по говорящим; связь технических параметров записи с меткой.
2) Предсказуемость личности: точность идентификации говорящего (leave-one-clip-out, 1-NN, косинусная
   близость) по (а) прокси формы лица и (б) статистическим дескрипторам мимики/позы — среди говорящих
   с >= 2 клипами. Высокая точность означает, что признаки несут информацию о личности, а значит
   при clip-level разбиении классификатор может опираться на «кто говорит», а не на «как говорит».
3) Потолок «идентификационного оракула»: точность, достижимую, зная только метки других клипов
   того же говорящего (большинство меток говорящего в обучении), при clip-level разбиении.

Запуск:  python src/audit.py --config configs/experiment.yaml   -> results/audit/*.json|csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, mannwhitneyu

import data

ROOT = Path(__file__).resolve().parents[1]


def loo_1nn_speaker_id(X, spk):
    keep = pd.Series(spk).map(pd.Series(spk).value_counts()).to_numpy() >= 2
    Xk, sk = X[keep], spk[keep]
    Xk = (Xk - Xk.mean(0)) / (Xk.std(0) + 1e-6)
    Xk = Xk / np.linalg.norm(Xk, axis=1, keepdims=True)
    S = Xk @ Xk.T
    np.fill_diagonal(S, -np.inf)
    pred = sk[S.argmax(1)]
    acc = float((pred == sk).mean())
    # «шанс»: вероятность совпадения при случайном выборе другого клипа
    counts = pd.Series(sk).value_counts()
    n = len(sk)
    chance = float(sum(c * (c - 1) for c in counts) / (n * (n - 1)))
    return dict(n_clips=int(n), n_speakers=int(counts.size), acc=acc, chance=chance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out = ROOT / cfg["results_dir"] / "audit"
    out.mkdir(parents=True, exist_ok=True)
    meta = data.load_meta()
    y = meta.label.to_numpy()
    spk = meta.speaker_id.to_numpy()
    rep = {}

    g = meta.groupby("speaker_id").label.agg(["count", "mean"])
    rep["dataset"] = dict(
        n_clips=len(meta), n_deceptive=int(y.sum()), n_truthful=int((1 - y).sum()),
        n_speakers=int(g.shape[0]), n_cases=int(meta.case_id.nunique()),
        speakers_both_labels=int(((g["mean"] > 0) & (g["mean"] < 1)).sum()),
        clips_from_both_label_speakers=int(g[(g["mean"] > 0) & (g["mean"] < 1)]["count"].sum()),
        max_clips_per_speaker=int(g["count"].max()), single_clip_speakers=int((g["count"] == 1).sum()),
        clips_in_top4_speakers=int(g["count"].sort_values(ascending=False).head(4).sum()),
        clips_in_top5_speakers=int(g["count"].sort_values(ascending=False).head(5).sum()),
        female_clips=int((meta.gender == "F").sum()), female_lie_share=float(y[meta.gender == "F"].mean()),
        male_clips=int((meta.gender == "M").sum()), male_lie_share=float(y[meta.gender == "M"].mean()),
        n_female_speakers=int(meta.drop_duplicates("speaker_id").gender.eq("F").sum()),
        n_male_speakers=int(meta.drop_duplicates("speaker_id").gender.eq("M").sum()),
    )
    g.sort_values("count", ascending=False).to_csv(out / "clips_per_speaker.csv")

    # Технические параметры записи vs метка
    res_str = meta.width.astype(str) + "x" + meta.height.astype(str)
    ct = pd.crosstab(res_str, meta.label)
    chi2, p, dof, _ = chi2_contingency(ct)
    rep["resolution_vs_label"] = dict(n_resolutions=int(ct.shape[0]), chi2=float(chi2), dof=int(dof), p=float(p),
                                      resolutions_single_label=int(((ct > 0).sum(axis=1) == 1).sum()),
                                      clips_in_single_label_resolutions=int(ct[(ct > 0).sum(axis=1) == 1].sum().sum()))
    ct.to_csv(out / "resolution_vs_label.csv")
    for col in ("duration", "fps"):
        u, pu = mannwhitneyu(meta.loc[y == 1, col], meta.loc[y == 0, col])
        rep[f"{col}_vs_label"] = dict(median_deceptive=float(meta.loc[y == 1, col].median()),
                                      median_truthful=float(meta.loc[y == 0, col].median()), U=float(u), p=float(pu))
    ex = json.load(open(data.FEAT_DIR / "extraction_meta.json"))["clips"]
    exd = pd.DataFrame(ex)
    exd["clip"] = exd["clip"].str.replace(".mp4", "", regex=False)
    exd = exd.set_index("clip").loc[meta["clip"]]
    rep["face_tracking"] = dict(valid_ratio_mean=float(exd.valid_ratio.mean()),
                                valid_ratio_min=float(exd.valid_ratio.min()),
                                clips_valid_below_0_5=int((exd.valid_ratio < 0.5).sum()),
                                median_face_px_median=float(exd.median_face_px.median()),
                                median_face_px_min=float(exd.median_face_px.min()))
    u, pu = mannwhitneyu(exd.valid_ratio[y == 1], exd.valid_ratio[y == 0])
    rep["face_tracking"]["valid_ratio_label_mannwhitney_p"] = float(pu)

    # Предсказуемость личности
    clips = meta["clip"].tolist()
    rep["speaker_id_sface"] = loo_1nn_speaker_id(data.sface_matrix(clips), spk)
    rep["speaker_id_face_shape"] = loo_1nn_speaker_id(data.identity_matrix(clips), spk)
    rep["speaker_id_stat_blend_pose"] = loo_1nn_speaker_id(data.stat_matrix(clips, cfg["feature_set"]), spk)
    rep["speaker_id_metadata"] = loo_1nn_speaker_id(data.metadata_matrix(meta), spk)

    # Оракул «метка большинства клипов того же говорящего в обучающей части» (clip-level, 5 seeds)
    accs = []
    for seed in cfg["seeds"]:
        fold = np.array(json.load(open(ROOT / f"splits/clip_seed{seed}.json"))["fold"])
        pred = np.zeros(len(y))
        for k in np.unique(fold):
            tr = fold != k
            prior = y[tr].mean()
            for i in np.where(fold == k)[0]:
                same = tr & (spk == spk[i])
                pred[i] = y[same].mean() if same.any() else prior
        accs.append(float(((pred >= 0.5) == y).mean()))
    rep["speaker_label_oracle_clip_protocol"] = dict(acc_mean=float(np.mean(accs)), acc_per_seed=accs)

    json.dump(rep, open(out / "audit.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
