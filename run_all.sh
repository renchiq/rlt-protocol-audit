#!/usr/bin/env bash
# Полное воспроизведение эксперимента: от архива RLT до таблиц и графиков.
# Запуск из корня репозитория:  bash run_all.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO"
mkdir -p logs data/raw data/models

# 1. Окружение (Python 3.12)
if [ ! -x "$PY" ]; then
  python3.12 -m venv "$REPO/.venv"
  "$REPO/.venv/bin/pip" install -r requirements.lock.txt
fi

# 2. Данные и модели (проверка контрольных сумм)
cd "$REPO/data/raw"
[ -f RLT.zip ] || curl -L -o RLT.zip https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip
[ -d Real-life_Deception_Detection_2016 ] || unzip -q RLT.zip -x "__MACOSX/*"
cd "$REPO/data/models"
[ -f face_landmarker.task ] || curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
[ -f face_detection_yunet_2023mar.onnx ] || curl -L -o face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
[ -f face_recognition_sface_2021dec.onnx ] || curl -L -o face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
cd "$REPO/data"
shasum -a 256 -c "$REPO/checksums.sha256"

# 3. Признаки (≈ 12 мин + 7 мин на Apple M5 Max, CPU)
cd "$REPO"
"$PY" src/detect_faces.py --config configs/extract.yaml 2>&1 | grep -v -E '^(W0000|I0000)' | tee logs/detect_faces.log
"$PY" src/extract_features.py --config configs/extract.yaml 2>&1 | grep -v -E '^(W0000|I0000)' | tee logs/extract_features.log

# 4. Метаданные, разбиения, аудит
"$PY" src/build_metadata.py
"$PY" src/splits.py --config configs/experiment.yaml
"$PY" src/audit.py --config configs/experiment.yaml | tee logs/audit.log

# 5. Модели (≈ 10 мин) и агрегация (≈ 5 мин)
rm -f logs/run_experiment.log results/metrics_raw.csv
"$PY" -W ignore src/run_experiment.py --config configs/experiment.yaml
"$PY" -W ignore src/count_params.py --config configs/experiment.yaml
"$PY" src/aggregate.py --config configs/aggregate.yaml | tee logs/aggregate.log
"$PY" -W ignore src/pair_analysis.py --config configs/aggregate.yaml | tee logs/pair_analysis.log
"$PY" src/figures.py --config configs/aggregate.yaml
