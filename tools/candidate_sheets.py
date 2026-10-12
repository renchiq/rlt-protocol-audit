"""Листы для ручной проверки выбора целевого лица: для каждого клипа — кадр из середины клипа и миниатюры
кандидатов c0..c3 (с числом кадров). Результат: data/derived/candidate_sheets/sheet_XX.jpg (не публикуется)."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = (ROOT / "data").resolve()
clips = sorted((D / "raw/Real-life_Deception_Detection_2016/Clips").glob("*/*.mp4"))
OUT = D / "derived/candidate_sheets"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for p in clips:
    z = np.load(D / f"derived/rlt_faces/{p.stem}.npz")
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, fr = cap.read()
    cap.release()
    fr = cv2.resize(fr, (int(112 * fr.shape[1] / fr.shape[0]), 112)) if ok else np.zeros((112, 200, 3), np.uint8)
    fr = cv2.resize(fr, (200, 112))
    tiles = [fr]
    for k in range(4):
        q = D / f"derived/rlt_candidates/{p.stem}_c{k}.jpg"
        t = cv2.resize(cv2.imread(str(q)), (112, 112)) if q.exists() else np.full((112, 112, 3), 40, np.uint8)
        if q.exists():
            cv2.putText(t, f"c{k} n={z['cand_n'][k]}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        tiles.append(t)
    lab = np.full((112, 120, 3), 255, np.uint8)
    cv2.putText(lab, p.stem.replace("trial_", ""), (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    rows.append(np.hstack([lab] + tiles))
for s in range(0, len(rows), 12):
    cv2.imwrite(str(OUT / f"sheet_{s // 12:02d}.jpg"), np.vstack(rows[s:s + 12]))
print(len(rows), "clips")
