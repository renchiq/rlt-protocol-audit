"""Сборка таблицы метаданных клипов RLT: метка, дело (trial), роль, технические параметры видео,
идентификатор говорящего (из ручной разметки metadata/speaker_annotation.csv).

Запуск:  python src/build_metadata.py
Результат: metadata/clips.csv
"""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Real-life_Deception_Detection_2016"


def parse_readme():
    rows = []
    for line in (DATA / "README.txt").read_text(errors="replace").splitlines():
        m = re.match(r"^\|\s*(trial_(lie|truth)_\d+)\.mp4\s*\|([^|]*)\|([^|]*)\|", line)
        if not m:
            continue
        role_case = m.group(3).strip()
        if "/" in role_case:
            role, case = [s.strip() for s in role_case.split("/", 1)]
        else:
            role, case = "", role_case
        rows.append(dict(clip=m.group(1), readme_role=role, readme_case=case, ruling=m.group(4).strip()))
    return pd.DataFrame(rows)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,bit_rate:format=duration,bit_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s, f = j["streams"][0], j["format"]
    num, den = s["avg_frame_rate"].split("/")
    return dict(width=int(s["width"]), height=int(s["height"]), fps=float(num) / float(den),
                duration=float(f["duration"]), bitrate=float(f.get("bit_rate", 0)))


def main():
    clips = sorted((DATA / "Clips").glob("*/*.mp4"))
    df = pd.DataFrame([dict(clip=p.stem, label=int("_lie_" in p.stem), **probe(p)) for p in clips])
    rd = parse_readme()
    # В README строка trial_truth_008 повторена дважды, а trial_truth_009 отсутствует:
    # вторую запись trial_truth_008 относим к trial_truth_009 (та же роль/дело).
    dup = rd.index[(rd["clip"] == "trial_truth_008").to_numpy()]
    if len(dup) == 2 and "trial_truth_009" not in set(rd["clip"]):
        rd.loc[dup[1], "clip"] = "trial_truth_009"
    df = df.merge(rd, on="clip", how="left")
    ann = pd.read_csv(ROOT / "metadata/speaker_annotation.csv")
    df = df.merge(ann, on="clip", how="left")
    assert df.speaker_id.notna().all(), df[df.speaker_id.isna()].clip.tolist()
    df["case_id"] = df["case_group"]
    df.to_csv(ROOT / "metadata/clips.csv", index=False)
    print(df.groupby("label").size())
    print("speakers:", df.speaker_id.nunique(), "cases:", df.case_id.nunique())
    print("speakers with both labels:", (df.groupby("speaker_id").label.nunique() > 1).sum())


if __name__ == "__main__":
    main()
