
"""Скрипт обучения, умеет работать с 4мя сценариями: бенчмарк, YOLO11s, её продолжение и YOLO11m."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from metrics import save_weighted_checkpoint


def train_stage(project_dir, stage):
    """Настройка обучения."""
    data_yaml = project_dir / "garbage.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Run prepare_data.py first: {data_yaml}")
    options = {
        "data": str(data_yaml), "imgsz": 960, "device": 0,
        "workers": 2, "amp": True, "cache": False, "seed": 42,
        "deterministic": True, "val": True, "save": True, "patience": 0,
        "hsv_h": 0.0, "project": str(project_dir / "runs"),
    }
    #этап бенчмарка
    if stage == "benchmark":
        weights = "yolo11s.pt"
        options.update(epochs=5, batch=16, close_mosaic=0, plots=False, name="speed_test_960")
    #обучение модели YOLO11s
    elif stage == "s":
        weights = "yolo11s.pt"
        options.update(epochs=50, batch=16, close_mosaic=0, plots=True, name="yolo11s_960_50ep")
    #дообучение модели YOLO11s
    elif stage == "s-finetune":
        weights = project_dir / "runs/yolo11s_960_50ep/weights/best_weighted.pt"
        options.update(
            epochs=100, batch=16, close_mosaic=10, plots=True,
            optimizer="AdamW", lr0=0.0001, lrf=0.01,
            momentum=0.9, warmup_epochs=1.0, warmup_bias_lr=0.0,
            name="yolo11s_960_finetune_100ep",
        )
    #обучение модели YOLO11m
    elif stage == "m":
        weights = "yolo11m.pt"
        options.update(
            epochs=100, batch=8, close_mosaic=0, plots=True,
            optimizer="AdamW", lr0=0.000526, momentum=0.9,
            weight_decay=0.0005, lrf=0.01, warmup_epochs=3.0,
            warmup_bias_lr=0.0, name="yolo11m_960_100ep",
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    run_dir = project_dir / "runs" / options["name"]
    if run_dir.exists():
        raise FileExistsError(f"Run directory exists; avoid an accidental second training: {run_dir}")
    if isinstance(weights, Path) and not weights.is_file():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights))
    if stage != "benchmark":
        model.add_callback("on_model_save", save_weighted_checkpoint)
    model.train(**options)


def main():
    """Запуск обучения, передаваемые параметры "--project" и "--stage"."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage", choices=["benchmark", "s", "s-finetune", "m"], required=True)
    args = parser.parse_args()
    train_stage(args.project.resolve(), args.stage)


if __name__ == "__main__":
    main()
