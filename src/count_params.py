"""Число обучаемых параметров нейросетевых моделей (results/model_params.json).
Запуск:  python src/count_params.py --config configs/experiment.yaml  -> results/model_params.json"""
import argparse
import json
from pathlib import Path

import yaml

import models

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
cfg = yaml.safe_load(open(ap.parse_args().config))
d_in = 55  # 52 blendshape + 3 угла (feature_set blend_pose)
out = {n: sum(p.numel() for p in models.build(n, d_in, cfg["deep"]).parameters() if p.requires_grad)
       for n in ("framepool", "gru", "transformer")}
json.dump(out, open(ROOT / "results/model_params.json", "w"), indent=1)
print(out)
