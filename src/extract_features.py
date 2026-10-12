"""Этап B: покадровые лицевые признаки целевого говорящего (MediaPipe Face Landmarker).

Использует результаты этапа A (src/detect_faces.py): лица и дескрипторы SFace на каждом прореженном
кадре и кандидатов целевого лица. Номер кандидата для клипа берётся из metadata/target_face.csv
(ручная проверка миниатюр кандидатов; по умолчанию 0 — кандидат с наибольшим рангом). На каждом кадре
выбирается лицо, наиболее похожее на целевое (косинус >= track_threshold), его квадратный кроп подаётся
в MediaPipe Face Landmarker. Сохраняются 52 blendshape-коэффициента, поза головы (yaw, pitch, roll,
tx, ty, tz), bbox, 478 ключевых точек (float16), маска валидных кадров, усреднённый дескриптор лица.

Запуск:  python src/extract_features.py --config configs/extract.yaml [--clips trial_lie_001,...]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml
from mediapipe.tasks.python import BaseOptions, vision

from detect_faces import read_frames


def rotation_to_euler(m):
    """Матрица поворота 3x3 -> (yaw, pitch, roll) в градусах."""
    sy = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(m[2, 1], m[2, 2]))
    yaw = np.degrees(np.arctan2(-m[2, 0], sy))
    roll = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    return yaw, pitch, roll


def square_crop(frame, box_px, scale, size):
    x, y, w, h = box_px
    s = max(w, h) * scale
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = x0 + int(round(s)), y0 + int(round(s))
    H, W = frame.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_CUBIC)


def process_clip(path, fz, cand, lm, cfg, crop_dir):
    times = fz["times"]
    T = len(times)
    W, H = int(fz["width"]), int(fz["height"])
    faces, embs = fz["faces"], fz["embs"]
    target = fz["cand_centers"][cand] if len(fz["cand_centers"]) > cand else None
    chosen = {}
    if target is not None and len(faces):
        sims = embs @ target
        for t_idx in np.unique(faces[:, 0]).astype(int):
            rows = np.where(faces[:, 0] == t_idx)[0]
            j = rows[np.argmax(sims[rows])]
            if sims[j] >= cfg["track_threshold"]:
                chosen[t_idx] = (j, float(sims[j]))

    blend = np.zeros((T, 52), np.float32)
    pose = np.zeros((T, 6), np.float32)
    bbox = np.zeros((T, 4), np.float32)
    lmk = np.zeros((T, 478, 3), np.float16)
    valid = np.zeros(T, bool)
    id_sim = np.zeros(T, np.float32)
    n_faces = (np.bincount(faces[:, 0].astype(int), minlength=T)[:T].astype(np.int8) if len(faces)
               else np.zeros(T, np.int8))
    names, used, crops = None, [], {}
    for i, frame in enumerate(read_frames(path, times)):
        if frame is None or i not in chosen:
            continue
        j, s = chosen[i]
        box = faces[j, 1:5]
        id_sim[i] = s
        bbox[i] = [(box[0] + box[2] / 2) / W, (box[1] + box[3] / 2) / H, box[2] / W, box[3] / H]
        crop = square_crop(frame, box, cfg["crop_scale"], cfg["crop_size"])
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            continue
        valid[i] = True
        used.append(j)
        bs = res.face_blendshapes[0]
        names = names or [c.category_name for c in bs]
        blend[i] = [c.score for c in bs]
        m = np.array(res.facial_transformation_matrixes[0])
        pose[i] = [*rotation_to_euler(m[:3, :3]), m[0, 3], m[1, 3], m[2, 3]]
        lmk[i] = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], np.float16)
        crops[i] = crop
    # контрольные кропы (25/50/75 % длительности; ближайший валидный кадр) для проверки разметки
    vidx = np.array(sorted(crops))
    crop_dir.mkdir(parents=True, exist_ok=True)
    for q in (25, 50, 75):
        if len(vidx):
            i = int(vidx[np.argmin(np.abs(vidx - T * q / 100))])
            cv2.imwrite(str(crop_dir / f"{path.stem}_{q}.jpg"), cv2.resize(crops[i], (160, 160)))
    id_emb = embs[used].mean(0) if used else np.zeros(128, np.float32)
    meta = dict(clip=path.name, candidate=int(cand), n_steps=T, src_fps=float(fz["src_fps"]), width=W, height=H,
                duration=float(fz["duration"]), valid_ratio=float(valid.mean()) if T else 0.0,
                tracked_ratio=len(chosen) / T if T else 0.0, multi_face_ratio=float((n_faces > 1).mean()),
                median_face_px=float(np.median(bbox[valid, 3]) * H) if valid.any() else 0.0)
    return dict(times=times, blend=blend, pose=pose, bbox=bbox, lmk=lmk, valid=valid, n_faces=n_faces,
                id_sim=id_sim, id_emb=id_emb.astype(np.float32)), names, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", help="через запятую; по умолчанию все")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = Path(args.config).resolve().parents[1]
    clips = sorted((root / cfg["clips_root"]).resolve().glob("*/*.mp4"))
    if args.clips:
        clips = [c for c in clips if c.stem in set(args.clips.split(","))]
    faces_dir = (root / cfg["faces_dir"]).resolve()
    out_dir = (root / cfg["features_dir"]).resolve()
    crop_dir = (root / cfg["crops_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = root / cfg["target_face_csv"]
    override = pd.read_csv(tf).set_index("clip")["candidate"].to_dict() if tf.exists() else {}
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str((root / cfg["model_path"]).resolve())),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True, min_face_detection_confidence=cfg["min_detection_conf"],
        min_face_presence_confidence=cfg["min_detection_conf"])
    meta_path = out_dir / "extraction_meta.json"
    old = json.load(open(meta_path)) if meta_path.exists() else {"clips": []}
    metas = {m["clip"]: m for m in old["clips"]}
    names = old.get("blendshape_names")
    t0 = time.time()
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for k, p in enumerate(clips):
            fz = np.load(faces_dir / f"{p.stem}.npz")
            feats, bn, meta = process_clip(p, fz, int(override.get(p.stem, 0)), lm, cfg, crop_dir)
            names = names or bn
            np.savez_compressed(out_dir / f"{p.stem}.npz", **feats)
            metas[p.name] = meta
            print(f"[{k + 1}/{len(clips)}] {p.name} cand={meta['candidate']} valid={meta['valid_ratio']:.2f} "
                  f"tracked={meta['tracked_ratio']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(blendshape_names=names, clips=[metas[c] for c in sorted(metas)], config=cfg,
                   mediapipe_version=mp.__version__, opencv_version=cv2.__version__),
              open(meta_path, "w"), indent=1)


if __name__ == "__main__":
    main()
