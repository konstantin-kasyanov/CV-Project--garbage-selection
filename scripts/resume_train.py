
"""Продолжение обучения в случае прерывания."""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from metrics import restore_weighted_checkpoint, save_weighted_checkpoint


def resume_run(run_dir):
    """Восстановление last.pt с сохранением лучшего взвешенного чекпоинта."""
    last = run_dir / "weights" / "last.pt"
    weighted_best = run_dir / "weights" / "best_weighted.pt"
    if not last.is_file() or not weighted_best.is_file():
        raise FileNotFoundError(f"Need both last.pt and best_weighted.pt in {run_dir}")
    backup = weighted_best.with_name("best_weighted_before_resume.pt")
    if not backup.exists():
        shutil.copy2(weighted_best, backup)
    model = YOLO(str(last))
    model.add_callback("on_train_start", restore_weighted_checkpoint)
    model.add_callback("on_model_save", save_weighted_checkpoint)
    model.train(resume=True)


def main():
    """Запуск восстановления обучения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    resume_run(args.run_dir.resolve())


if __name__ == "__main__":
    main()
