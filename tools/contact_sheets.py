"""Контрольные листы кропов лиц, сгруппированные по делу (из README датасета), для ручной разметки говорящих.
Каждая строка листа — клип (кропы на 25/50/75 % длительности). Результат: data/derived/contact_sheets/*.jpg
(не публикуется: содержит изображения из датасета)."""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_metadata import parse_readme  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CROPS = (ROOT / "data/derived/rlt_face_crops").resolve()
OUT = (ROOT / "data/derived/contact_sheets").resolve()
OUT.mkdir(parents=True, exist_ok=True)
rd = parse_readme()
dup = rd.index[(rd["clip"] == "trial_truth_008").to_numpy()]
rd.loc[dup[1], "clip"] = "trial_truth_009"
for case, g in rd.groupby("readme_case"):
    rows = []
    for c in g["clip"]:
        tiles = []
        for q in (25, 50, 75):
            p = CROPS / f"{c}_{q}.jpg"
            im = cv2.imread(str(p)) if p.exists() else np.zeros((160, 160, 3), np.uint8)
            tiles.append(cv2.resize(im, (120, 120)))
        lab = np.full((120, 150, 3), 255, np.uint8)
        cv2.putText(lab, c.replace("trial_", ""), (3, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(np.hstack([lab] + tiles))
    sheet = np.vstack(rows)
    name = re.sub(r"[^A-Za-z0-9]+", "_", case).strip("_")
    cv2.imwrite(str(OUT / f"{name}.jpg"), sheet)
print(len(list(OUT.glob("*.jpg"))), "sheets")
