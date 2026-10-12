"""Этап A: детекция лиц (YuNet) и дескрипторы лиц (SFace) на всех прореженных кадрах; кластеризация лиц
клипа в кандидатов «целевого говорящего».

Кандидаты строятся жадно: медоид самого плотного кластера дескрипторов (косинус >= id_threshold),
затем его члены исключаются и процедура повторяется. Кандидаты ранжируются по
(число кадров) x (средняя уверенность детектора) x (медианная высота лица). Для каждого кандидата
сохраняется миниатюра; выбор кандидата проверяется вручную (metadata/target_face.csv).

Запуск:  python src/detect_faces.py --config configs/extract.yaml
Результат: data/derived/rlt_faces/<clip>.npz, data/derived/rlt_candidates/<clip>_c<k>.jpg
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_frames(path, times):
    cap = cv2.VideoCapture(str(path))
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        yield frame if ok else None
    cap.release()


def clip_times(path, fps):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n_src / src_fps
    return np.arange(0, duration - 1e-6, 1.0 / fps), dict(src_fps=src_fps, n_src_frames=n_src, width=w,
                                                        height=h, duration=duration)


def cluster_candidates(embs, faces, thr, max_cand):
    left = np.ones(len(embs), bool)
    cands = []
    S = embs @ embs.T
    while left.sum() > 0 and len(cands) < max_cand:
        sup = ((S >= thr) & left[None, :] & left[:, None]).sum(1)
        sup[~left] = -1
        k = int(np.argmax(sup))
        members = left & (S[k] >= thr)
        c = embs[members].mean(0)
        c /= np.linalg.norm(c) + 1e-9
        score = members.sum() * faces[members, 14].mean() * np.median(faces[members, 3])
        cands.append(dict(center=c, n=int(members.sum()), score=float(score), medoid=k))
        left &= ~members
    cands.sort(key=lambda d: -d["score"])
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = Path(args.config).resolve().parents[1]
    clips = sorted((root / cfg["clips_root"]).resolve().glob("*/*.mp4"))
    out = (root / cfg["faces_dir"]).resolve()
    cand_dir = (root / cfg["candidates_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    det = cv2.FaceDetectorYN.create(str((root / cfg["detector_path"]).resolve()), "", (320, 320), cfg["detector_score"])
    rec = cv2.FaceRecognizerSF.create(str((root / cfg["recognizer_path"]).resolve()), "")
    t0 = time.time()
    for n, path in enumerate(clips):
        times, info = clip_times(path, cfg["fps"])
        rows, embs, thumbs = [], [], []
        for i, frame in enumerate(read_frames(path, times)):
            if frame is None:
                continue
            h, w = frame.shape[:2]
            det.setInputSize((w, h))
            _, fl = det.detect(frame)
            for f in ([] if fl is None else fl):
                if f[3] < cfg["min_face_px"]:
                    continue
                aligned = rec.alignCrop(frame, f)
                e = rec.feature(aligned).reshape(-1)
                embs.append(e / (np.linalg.norm(e) + 1e-9))
                rows.append(np.concatenate([[i], f]))
                thumbs.append(aligned)
        faces = np.array(rows, np.float32).reshape(-1, 16)  # t_idx, x, y, w, h, 5 точек (10), score
        embs = np.array(embs, np.float32).reshape(-1, 128)
        cands = cluster_candidates(embs, faces[:, 1:], cfg["id_threshold"], cfg["max_candidates"]) if len(embs) else []
        for k, c in enumerate(cands):
            cv2.imwrite(str(cand_dir / f"{path.stem}_c{k}.jpg"), thumbs[c["medoid"]])
        np.savez_compressed(out / f"{path.stem}.npz", faces=faces, embs=embs, times=times.astype(np.float32),
                            cand_centers=np.array([c["center"] for c in cands], np.float32).reshape(-1, 128),
                            cand_n=np.array([c["n"] for c in cands]), cand_score=np.array([c["score"] for c in cands]),
                            **{k: np.array(v) for k, v in info.items()})
        print(f"[{n + 1}/{len(clips)}] {path.name} faces={len(faces)} cands="
              + ",".join(f"{c['n']}" for c in cands) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
