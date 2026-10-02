
"""Инференс модели, пишет по образцу файла сабмишена в CSV, работает порционно, чтобы не грузить весь датасет в ОЗУ."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


def to_submission_objects(result):
    """Обратное преобразование результатов YOLO из нормализованных координат и 0+ классов в координаты xywh и 1+ классы."""
    height, width = result.orig_shape
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.int().cpu().numpy()
    objects = []
    for box, score, cls in zip(boxes, scores, classes):
        x1, x2 = np.clip(box[[0, 2]], 0, width)
        y1, y2 = np.clip(box[[1, 3]], 0, height)
        if x2 > x1 and y2 > y1:
            objects.append([
                int(cls + 1), float(score), float(x1), float(y1),
                float(x2 - x1), float(y2 - y1),
            ])
    return objects


def predict_test(project_dir, images_dir, weights, output, batch_size, tta):
    """Инференс модели в малых пакетах GPU и запись одной строки на изображение из test.csv."""
    if not weights.is_file():
        raise FileNotFoundError(weights)
    test = pd.read_csv(project_dir / "test.csv")
    if not test.file_name.is_unique:
        raise ValueError("test.csv contains duplicate filenames")
    names = test.file_name.tolist()
    paths = [images_dir / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])
    model = YOLO(str(weights))
    found = {}
    for start in range(0, len(paths), batch_size):
        results = model.predict(
            source=[str(p) for p in paths[start:start + batch_size]],
            stream=True, imgsz=960, batch=batch_size, device=0,
            conf=0.001, iou=0.7, max_det=300, augment=tta,
            quantize=32, save=False, verbose=False,
        )
        for result in results:
            found[Path(result.path).name] = to_submission_objects(result)
        if start == 0 and model.predictor.device.type != "cuda":
            raise RuntimeError("GPU prediction was requested, but predictor runs on CPU")
        if start % (25 * batch_size) == 0:
            print(f"{len(found)}/{len(names)} images")
    if set(found) != set(names):
        raise RuntimeError("Prediction count differs from test.csv")
    submission = pd.DataFrame({
        "file_name": names,
        "bbox": [json.dumps(found[name], allow_nan=False) for name in names],
    })
    submission.to_csv(output, index=False)
    check = pd.read_csv(output)
    if check.file_name.tolist() != names or check.bbox.isna().any():
        raise RuntimeError("Saved submission failed file order/format checks")
    print("Saved", output, "rows", len(check))
    del model
    torch.cuda.empty_cache()


def main():
    """Чтение моделей и целевых файлов из CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    predict_test(
        args.project.resolve(), args.images.resolve(), args.weights.resolve(),
        args.output.resolve(), args.batch, args.tta,
    )


if __name__ == "__main__":
    main()
