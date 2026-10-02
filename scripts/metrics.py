
"""Имплементация взвешенной метрики в модель."""

import csv
import shutil

import numpy as np


def weighted_map_from_ultralytics(metrics):
    """Рассчет взвешенного AP@[.50:.95] на валидации для каждого класса."""
    #достаем из объекта metrics (стандартный отчет валидации YOLO) три массива - id класса, езначение АР класса и количество объектов класса
    class_ids = np.asarray(metrics.box.ap_class_index, dtype=int)
    class_ap = np.asarray(metrics.box.ap, dtype=float)
    counts = np.asarray(metrics.nt_per_class, dtype=float)
    if len(class_ids) != len(class_ap) or not len(class_ap):
        raise ValueError("Validation AP and class IDs have incompatible shapes")
    #отбираем веса для классов, попавших в валидацию
    support = counts[class_ids]
    if support.sum() <= 0 or not np.isfinite(class_ap).all():
        raise ValueError("Validation AP or class support is invalid")
    return float(np.average(class_ap, weights=support))


def save_weighted_checkpoint(trainer):
    """Отслеживаем взвешенную метрику после каждой эпохи, записываем историю в CSV-файл и 
    сохраняем веса модели (.pt) на основе лучшего значения именно этой взвешенной метрики."""
    #считаем метрику
    metrics = trainer.validator.metrics
    score = weighted_map_from_ultralytics(metrics)
    #создаем файл и/или добавляем актуальную эпоху
    history = trainer.save_dir / "weighted_metrics.csv"
    new_file = not history.exists()
    with history.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(["epoch", "weighted_map50_95", "map50_95"])
        writer.writerow([trainer.epoch + 1, score, float(metrics.box.map)])
    #ищем лучшую эпоху по взвешенной метрике
    if score > getattr(trainer, "_best_weighted_map", -float("inf")):
        trainer._best_weighted_map = score
        trainer._best_weighted_epoch = trainer.epoch + 1

        shutil.copy2(trainer.last, trainer.save_dir / "weights" / "best_weighted.pt")
    print(
        f"weighted mAP: {score:.6f}; best {trainer._best_weighted_map:.6f} "
        f"at epoch {trainer._best_weighted_epoch}",
        flush=True,
    )


def restore_weighted_checkpoint(trainer):
    """На случай прерывания обучения восстанавливаем лучшую метрику предыдущих эпох."""
    history = trainer.save_dir / "weighted_metrics.csv"
    with history.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No completed epochs in {history}")
    best = max(rows, key=lambda row: float(row["weighted_map50_95"]))
    trainer._best_weighted_map = float(best["weighted_map50_95"])
    trainer._best_weighted_epoch = int(best["epoch"])
