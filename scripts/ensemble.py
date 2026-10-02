
"""Сравнение двух моделей и ансамбля WBF на валидации, объединение тестовых предсказаний из CSV."""

import argparse
import ast
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO


def validation_paths(project_dir):
    """Чтение путей к изображениям из фиксированного валидационного набора и проверка их наличия."""
    paths = [Path(line.strip()) for line in (project_dir / "val_images.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths or not all(p.is_file() for p in paths):
        raise FileNotFoundError("val_images.txt contains missing image paths")
    return paths


def targets_from_csv(project_dir, paths):
    """Подготовка истинной разметки(xywh в xyxy и классы в 0+)."""
    annotations = pd.read_csv(project_dir / "train.csv").dropna(subset=["bbox", "category_id"])
    groups = {name: rows for name, rows in annotations.groupby("file_name")}
    targets = []
    for path in paths:
        rows = groups.get(path.name)
        if rows is None:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty(0, dtype=np.int64)
        else:
            boxes = np.array([ast.literal_eval(value) for value in rows.bbox], dtype=np.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            labels = rows.category_id.to_numpy(dtype=np.int64) - 1
        targets.append({"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels)})
    return targets


def predict_validation(weights, paths):
    """Получение предсказаний модели с TTA на валидационном наборе небольшими батчами."""
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights))
    predictions = {}
    for start in range(0, len(paths), 4):
        results = model.predict(
            source=[str(path) for path in paths[start:start + 4]],
            stream=True, imgsz=960, batch=4, device=0, augment=True,
            quantize=32, conf=0.001, iou=0.7, max_det=300, verbose=False,
        )
        for result in results:
            predictions[Path(result.path).name] = {
                "boxes": result.boxes.xyxy.cpu().numpy(),
                "scores": result.boxes.conf.cpu().numpy(),
                "labels": result.boxes.cls.int().cpu().numpy(),
                "shape": result.orig_shape,
            }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def fuse_frame(first, second, iou, scale):
    """Нормировка рамок, объединение предсказаний двух моделей через WBF и возврат координат в пикселях."""
    parts = []
    for pred in (first, second):
        boxes = np.clip(pred["boxes"] / scale, 0, 1)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        parts.append((boxes[keep].tolist(), pred["scores"][keep].tolist(), pred["labels"][keep].tolist()))
    boxes, scores, labels = weighted_boxes_fusion(
        [part[0] for part in parts],
        [part[1] for part in parts],
        [part[2] for part in parts],
        weights=[1, 1], iou_thr=iou, skip_box_thr=0.001,
    )
    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4) * scale,
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def weighted_map(predictions, paths, targets):
    """Расчёт mAP по порогам IoU 0.50–0.95 с весами по количеству объектов каждого класса."""
    metric = MeanAveragePrecision(
        box_format="xyxy", class_metrics=True, backend="faster_coco_eval",
        max_detection_thresholds=[1, 10, 100],
    )
    for path, target in zip(paths, targets):
        pred = predictions[path.name]
        metric.update([{
            "boxes": torch.as_tensor(pred["boxes"], dtype=torch.float32),
            "scores": torch.as_tensor(pred["scores"], dtype=torch.float32),
            "labels": torch.as_tensor(pred["labels"], dtype=torch.int64),
        }], [target])
    values = metric.compute()
    support = np.bincount(np.concatenate([target["labels"].numpy() for target in targets]), minlength=15)
    class_ids = values["classes"].numpy().astype(int)
    return float(np.average(values["map_per_class"].numpy(), weights=support[class_ids]))


def evaluate_ensemble(project_dir, weights_s, weights_m, thresholds):
    """Сравнение отдельных моделей и ансамбля WBF при разных порогах IoU на одинаковых предсказаниях."""
    paths = validation_paths(project_dir)
    targets = targets_from_csv(project_dir, paths)
    pred_s = predict_validation(weights_s, paths)
    pred_m = predict_validation(weights_m, paths)
    scores = {"s": weighted_map(pred_s, paths, targets), "m": weighted_map(pred_m, paths, targets)}
    for threshold in thresholds:
        fused = {}
        for path in paths:
            name = path.name
            if pred_s[name]["shape"] != pred_m[name]["shape"]:
                raise ValueError(f"Different image sizes for {name}")
            height, width = pred_s[name]["shape"]
            scale = np.array([width, height, width, height], dtype=np.float32)
            fused[name] = fuse_frame(pred_s[name], pred_m[name], threshold, scale)
        scores[f"wbf_iou_{threshold:g}"] = weighted_map(fused, paths, targets)
    reports = project_dir / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "ensemble_validation.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(scores, indent=2, ensure_ascii=False))


def parse_csv_box_list(value, scale):
    """Чтение списка объектов из ячейки CSV и перевод пиксельных xywh в нормированные xyxy."""
    if pd.isna(value):
        raise ValueError("bbox cell is missing; use '[]' for a frame without objects")
    rows = np.asarray(ast.literal_eval(value), dtype=float).reshape(-1, 6)
    boxes = rows[:, 2:6].copy()
    boxes[:, 2:] += boxes[:, :2]
    boxes = np.clip(boxes / scale, 0, 1)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], rows[keep, 1], rows[keep, 0].astype(int)


def fuse_test_csvs(project_dir, images_dir, csv_s, csv_m, output, iou):
    """Объединение двух тестовых CSV через WBF и сохранение рамок в формате xywh в порядке test.csv."""
    test = pd.read_csv(project_dir / "test.csv")
    s = pd.read_csv(csv_s).set_index("file_name")["bbox"]
    m = pd.read_csv(csv_m).set_index("file_name")["bbox"]
    if not test.file_name.is_unique or not s.index.is_unique or not m.index.is_unique:
        raise ValueError("Duplicate test filenames")
    if set(s.index) != set(m.index) or set(s.index) != set(test.file_name):
        raise ValueError("The prediction CSVs do not cover all files in test.csv")
    rows = []
    for name in test.file_name:
        with Image.open(images_dir / name) as image:
            width, height = image.size
        scale = np.asarray([width, height, width, height], dtype=np.float32)
        parts = [parse_csv_box_list(s.loc[name], scale), parse_csv_box_list(m.loc[name], scale)]
        boxes, scores, labels = weighted_boxes_fusion(
            [part[0].tolist() for part in parts],
            [part[1].tolist() for part in parts],
            [part[2].tolist() for part in parts],
            weights=[1, 1], iou_thr=iou, skip_box_thr=0.001,
        )
        pixel_xywh = np.asarray(boxes).reshape(-1, 4) * scale
        pixel_xywh[:, 2:] -= pixel_xywh[:, :2]
        objects = [
            [int(label), round(float(score), 8), *[round(float(x), 4) for x in box]]
            for label, score, box in zip(labels, scores, pixel_xywh)
        ]
        rows.append((name, json.dumps(objects, allow_nan=False)))
    pd.DataFrame(rows, columns=["file_name", "bbox"]).to_csv(output, index=False)
    saved = pd.read_csv(output)
    if saved.file_name.tolist() != test.file_name.tolist():
        raise RuntimeError("Saved submission is missing files or has wrong order")
    print(f"Saved {len(saved)} rows: {output}")


def main():
    """Чтение аргументов командной строки и запуск валидации или формирования итогового CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["val", "submit"], required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--weights-s", type=Path)
    parser.add_argument("--weights-m", type=Path)
    parser.add_argument("--csv-s", type=Path)
    parser.add_argument("--csv-m", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou", type=float, nargs="+", required=True)
    args = parser.parse_args()
    project_dir = args.project.resolve()
    if args.mode == "val":
        if args.weights_s is None or args.weights_m is None:
            parser.error("val mode needs --weights-s and --weights-m")
        evaluate_ensemble(project_dir, args.weights_s.resolve(), args.weights_m.resolve(), args.iou)
    else:
        if args.images is None or args.csv_s is None or args.csv_m is None or args.output is None:
            parser.error("submit mode needs --images, --csv-s, --csv-m and --output")
        if len(args.iou) != 1:
            parser.error("submit mode needs exactly one selected --iou value")
        fuse_test_csvs(
            project_dir, args.images.resolve(), args.csv_s.resolve(),
            args.csv_m.resolve(), args.output.resolve(), args.iou[0],
        )


if __name__ == "__main__":
    main()
