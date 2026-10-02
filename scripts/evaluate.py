
"""Оценка качества модели на лучшей итерации через взвешенную mAP, macro mAP и mAP50."""

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from metrics import weighted_map_from_ultralytics


def evaluate_model(project_dir, weights, imgsz, tta):
    """Запускаем валидацию Ultralytics и рассчитываем метрики."""
    if not weights.is_file():
        raise FileNotFoundError(weights)
    name = f"{weights.parent.parent.name}_{imgsz}_{'tta' if tta else 'plain'}"
    metrics = YOLO(str(weights)).val(
        data=str(project_dir / "garbage.yaml"), split="val", imgsz=imgsz,
        batch=4, device=0, workers=0, quantize=32,
        conf=0.001, iou=0.7, max_det=300,
        rect=True, augment=tta, plots=False, save_json=False,
        project=str(project_dir / "runs"), name=f"validation_{name}",
    )
    result = {
        "checkpoint": str(weights), "imgsz": imgsz, "tta": tta,
        "weighted_map50_95": weighted_map_from_ultralytics(metrics),
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "val_class_ids": np.asarray(metrics.box.ap_class_index, dtype=int).tolist(),
    }
    output_dir = project_dir / "reports"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"validation_{name}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    """Парсим чекпоинт, размер изображения и флаг TTA из CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    evaluate_model(args.project.resolve(), args.weights.resolve(), args.imgsz, args.tta)


if __name__ == "__main__":
    main()
