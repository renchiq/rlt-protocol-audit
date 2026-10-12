"""Модели над последовательностями покадровых признаков (PyTorch)."""
import math

import numpy as np
import torch
from torch import nn


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TransformerClassifier(nn.Module):
    """Энкодер Transformer над временной последовательностью, маскированное усреднение, линейная голова."""

    def __init__(self, d_in, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.3):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        h = self.pos(self.inp(x))
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.head(self.drop(masked_mean(h, mask))).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, d_in, hidden=32, dropout=0.3):
        super().__init__()
        self.inp = nn.Linear(d_in, 64)
        self.gru = nn.GRU(64, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, mask):
        h, _ = self.gru(torch.relu(self.inp(x)))
        return self.head(self.drop(masked_mean(h, mask))).squeeze(-1)


class FramePoolClassifier(nn.Module):
    """Порядко-инвариантный контроль: покадровый MLP + усреднение и std по времени."""

    def __init__(self, d_in, hidden=64, dropout=0.3):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, mask):
        h = self.frame(x)
        mu = masked_mean(h, mask)
        var = masked_mean((h - mu.unsqueeze(1)) ** 2, mask)
        return self.head(self.drop(torch.cat([mu, var.clamp(min=1e-6).sqrt()], -1))).squeeze(-1)


def build(name, d_in, cfg):
    if name.startswith("transformer"):
        return TransformerClassifier(d_in, **cfg.get("transformer", {}))
    if name.startswith("gru"):
        return GRUClassifier(d_in, **cfg.get("gru", {}))
    if name.startswith("framepool"):
        return FramePoolClassifier(d_in, **cfg.get("framepool", {}))
    raise ValueError(name)


def _windows(x, L, stride):
    T = len(x)
    if T <= L:
        return [x]
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] != T - L:
        starts.append(T - L)
    return [x[s: s + L] for s in starts]


def _pad(batch, L, d):
    xb = np.zeros((len(batch), L, d), np.float32)
    mb = np.zeros((len(batch), L), bool)
    for i, s in enumerate(batch):
        xb[i, : len(s)] = s
        mb[i, : len(s)] = True
    return torch.from_numpy(xb), torch.from_numpy(mb)


def _predict(model, seqs, L, stride, d):
    model.eval()
    probs = []
    with torch.no_grad():
        for s in seqs:
            w = _windows(s, L, stride)
            xb, mb = _pad(w, L, d)
            probs.append(float(torch.sigmoid(model(xb, mb)).mean()))
    return np.array(probs)


def _fit(name, train_seqs, y_train, cfg, seed, epochs, val=None, rng=None):
    """Обучение с фиксированными гиперпараметрами. Если задан val = (seqs, y), после каждой эпохи
    считается BCE на валидационных клипах (оценка клипа — среднее по окнам, как в тесте)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed) if rng is None else rng
    tc = cfg["train"]
    L = tc["window"]
    shuffled = name.endswith("_shuffled")
    allf = np.concatenate(train_seqs)
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    tr = [(s - mu) / sd for s in train_seqs]
    d = tr[0].shape[1]
    model = build(name, d, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    y_train = np.asarray(y_train, np.float32)
    n = len(tr)
    val_loss = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        for b in range(0, n, tc["batch_size"]):
            ids = order[b: b + tc["batch_size"]]
            batch = []
            for i in ids:
                s = tr[i]
                if shuffled:
                    s = s[rng.permutation(len(s))]
                if len(s) > L:
                    st = rng.integers(0, len(s) - L + 1)
                    s = s[st: st + L]
                if tc.get("noise_std", 0) > 0:
                    s = s + rng.normal(0, tc["noise_std"], s.shape).astype(np.float32)
                batch.append(s)
            xb, mb = _pad(batch, L, d)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb, mb), torch.from_numpy(y_train[ids]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val is not None:
            pv = np.clip(_predict(model, [(s - mu) / sd for s in val[0]], L, tc["eval_stride"], d), 1e-6, 1 - 1e-6)
            yv = np.asarray(val[1], float)
            val_loss.append(float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))))
    return model, (mu, sd), float(loss.detach()), val_loss


def train_and_predict(name, train_seqs, y_train, test_seqs, cfg, seed, inner=None):
    """Обучение с фиксированными гиперпараметрами (без подбора по тесту), предсказание вероятностей.

    Для варианта *_shuffled порядок кадров случайно переставляется (в обучении — заново на каждой эпохе,
    в тесте — фиксированной перестановкой), что разрушает временную структуру при сохранении
    распределения кадров.

    Для варианта *_es (ранняя остановка, проверка устойчивости) число эпох выбирается по минимуму BCE
    на внутренней валидационной части обучающей выборки: inner = (fit_idx, val_idx) — разбиение
    обучающих клипов того же типа, что и внешнее (по клипам / говорящим / делам). Затем модель заново
    обучается на всей обучающей части выбранное число эпох. Тестовый фолд в выборе не участвует.
    """
    tc = cfg["train"]
    L, stride = tc["window"], tc["eval_stride"]
    rng = np.random.default_rng(seed)
    info = {}
    epochs = tc["epochs"]
    if name.endswith("_es"):
        fit_idx, val_idx = inner
        _, _, _, vl = _fit(name, [train_seqs[i] for i in fit_idx], np.asarray(y_train)[fit_idx], cfg, seed,
                           tc["epochs"], val=([train_seqs[i] for i in val_idx], np.asarray(y_train)[val_idx]))
        epochs = int(np.argmin(vl)) + 1
        info = dict(best_epoch=epochs, val_loss_best=round(min(vl), 4), val_loss_last=round(vl[-1], 4))
    if name.endswith("_shuffled"):
        # перестановка тестовых кадров до обучения — тот же порядок вызовов генератора, что в исходном запуске
        test_seqs = [s[rng.permutation(len(s))] for s in test_seqs]
    model, (mu, sd), last_loss, _ = _fit(name, train_seqs, y_train, cfg, seed, epochs, rng=rng)
    te = [(s - mu) / sd for s in test_seqs]
    return _predict(model, te, L, stride, te[0].shape[1]), last_loss, info
