"""Загрузка покадровых признаков, построение последовательностей и статистических дескрипторов."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = (ROOT / "data/derived/rlt_mediapipe_10fps").resolve()
GESTURE_CSV = (ROOT / "data/raw/Real-life_Deception_Detection_2016/Annotation/"
               "All_Gestures_Deceptive and Truthful.csv").resolve()


def load_meta():
    return pd.read_csv(ROOT / "metadata/clips.csv")


def blendshape_names():
    return json.load(open(FEAT_DIR / "extraction_meta.json"))["blendshape_names"]


def frame_features(clip, feature_set="blend_pose"):
    """Покадровая матрица признаков [T, D] и маска кадров с лицом.

    blend_pose: 52 blendshape + yaw/pitch/roll (55)
    blend:      только 52 blendshape
    pose:       только yaw/pitch/roll
    """
    z = np.load(FEAT_DIR / f"{clip}.npz")
    blend, pose, valid = z["blend"], z["pose"], z["valid"]
    angles = pose[:, :3] / 30.0  # градусы -> порядок единиц
    if feature_set == "blend_pose":
        x = np.concatenate([blend, angles], 1)
    elif feature_set == "blend":
        x = blend
    elif feature_set == "pose":
        x = angles
    else:
        raise ValueError(feature_set)
    return x.astype(np.float32), valid


def fill_invalid(x, valid):
    """Кадры без лица заполняются последним валидным значением (в начале — первым валидным)."""
    if valid.all() or not valid.any():
        return x
    x = x.copy()
    idx = np.where(valid, np.arange(len(valid)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(valid)
    idx[idx < 0] = first
    return x[idx]


def load_sequences(clips, feature_set="blend_pose"):
    seqs, valids = [], []
    for c in clips:
        x, v = frame_features(c, feature_set)
        seqs.append(fill_invalid(x, v))
        valids.append(v)
    return seqs, valids


def stat_descriptor(x, valid):
    """Порядко-инвариантные функционалы по валидным кадрам: mean, std, p10, p90 и
    средний модуль первой разности (единственный признак, зависящий от соседства кадров)."""
    xv = x[valid] if valid.sum() >= 2 else x
    d = np.abs(np.diff(x, axis=0))
    dv = d[valid[1:] & valid[:-1]] if (valid[1:] & valid[:-1]).sum() >= 1 else d
    return np.concatenate([xv.mean(0), xv.std(0), np.percentile(xv, 10, 0), np.percentile(xv, 90, 0),
                           dv.mean(0) if len(dv) else np.zeros(x.shape[1])])


def stat_matrix(clips, feature_set="blend_pose", include_delta=True):
    seqs, valids = load_sequences(clips, feature_set)
    X = np.stack([stat_descriptor(s, v) for s, v in zip(seqs, valids)])
    if not include_delta:
        X = X[:, : 4 * seqs[0].shape[1]]
    return X


def gesture_matrix(clips):
    g = pd.read_csv(GESTURE_CSV)
    g["clip"] = g["id"].str.replace(".mp4", "", regex=False)
    g = g.set_index("clip").drop(columns=["id", "class"])
    return g.loc[list(clips)].to_numpy(np.float32), list(g.columns)


def metadata_matrix(meta):
    """Технические параметры записи (аудит «коротких путей»): разрешение, fps, длительность, битрейт."""
    return np.stack([meta.width, meta.height, meta.fps, meta.duration, np.log1p(meta.bitrate)], 1).astype(np.float32)


def identity_matrix(clips):
    """Прокси идентичности: усреднённая нормированная 3D-форма лица (ключевые точки MediaPipe),
    центрированная и масштабированная по межзрачковому расстоянию. Не содержит динамики."""
    out = []
    for c in clips:
        z = np.load(FEAT_DIR / f"{c}.npz")
        l = z["lmk"][z["valid"]].astype(np.float32)
        if len(l) == 0:
            l = z["lmk"].astype(np.float32)
        l = l - l.mean(1, keepdims=True)
        scale = np.linalg.norm(l[:, 468] - l[:, 473], axis=1, keepdims=True)[:, :, None] + 1e-6
        l = l / scale
        out.append(np.median(l, 0).reshape(-1))
    return np.stack(out)


def sface_matrix(clips):
    """Усреднённый по валидным кадрам дескриптор лица SFace (распознавание лиц) — явный носитель личности."""
    return np.stack([np.load(FEAT_DIR / f"{c}.npz")["id_emb"] for c in clips])
