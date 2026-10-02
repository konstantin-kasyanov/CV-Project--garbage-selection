
"""Преобразование датасета в YOLO TXT под требования модели(нормализация координат, первый класс имеет метку 0)."""

import argparse
import ast
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split


CLASS_NAMES = {
    1: "PET (transparent) (green)",
    2: "PET (transparent) (brown)",
    3: "PET (transparent) (blue)",
    4: "PET (transparent)",
    5: "PET (transparent) (dark blue)",
    6: "PET (black)",
    7: "PET (white)",
    8: "PET (sticker)",
    9: "PET (flacon)",
    10: "PET (household chemicals)",
    11: "PND (household chemicals)",
    12: "PND packet",
    13: "Other plastic",
    14: "Other plastic (transparent)",
    15: "Not plastic",
}


def convert_box(bbox, category_id, image_width, image_height):
    """Нормализация координат из пиксельной разметки, коррекция меток (начало с 0)."""
    x, y, width, height = ast.literal_eval(bbox)
    cls = int(category_id) - 1
    if not 0 <= cls < 15 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid annotation: {bbox}, class {category_id}")
    if x < 0 or y < 0 or x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
        raise ValueError(f"Bounding box outside image: {bbox}")
    return (
        f"{cls} {(x + width / 2) / image_width:.8f} "
        f"{(y + height / 2) / image_height:.8f} "
        f"{width / image_width:.8f} {height / image_height:.8f}"
    )


def prepare_dataset(project_dir, images_dir):
    """Проверяем директории с фото, создаем выборки train/val, готовим папку labels"""
    #проверка названия директорий под требования YOLO и соответствия перечня и фактических фотографий
    if images_dir.name != "images":
        raise ValueError("The JPEG folder must be named 'images' for YOLO label lookup")
    annotations = pd.read_csv(project_dir / "train.csv")
    image_names = sorted(annotations.file_name.unique())
    missing = [name for name in image_names if not (images_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing labeled images: {missing[:5]}")
    #делим выборки
    train_names, val_names = train_test_split(image_names, test_size=0.2, random_state=42)
    #Генерация YOLO-разметки - в labels записаны нормализованные координаты объектов к каждому jpeg, делаем txt с выборками
    labels_dir = images_dir.parent / "labels"
    labels_dir.mkdir(exist_ok=True)

    count_objects = 0
    count_empty = 0
    for filename, rows in annotations.groupby("file_name"):
        with Image.open(images_dir / filename) as image:
            width, height = image.size
        labels = [
            convert_box(row.bbox, row.category_id, width, height)
            for row in rows.dropna(subset=["bbox", "category_id"]).itertuples()
        ]
        (labels_dir / Path(filename).with_suffix(".txt")).write_text(
            "\n".join(labels), encoding="utf-8"
        )
        count_objects += len(labels)
        count_empty += not labels

    for split, names in (("train", train_names), ("val", val_names)):
        image_paths = [(images_dir / name).as_posix() for name in names]
        (project_dir / f"{split}_images.txt").write_text(
            "\n".join(image_paths) + "\n", encoding="utf-8"
        )
    #готовим garbage.yaml
    config = {
        "path": project_dir.as_posix(),
        "train": (project_dir / "train_images.txt").as_posix(),
        "val": (project_dir / "val_images.txt").as_posix(),
        "names": {cls - 1: name for cls, name in CLASS_NAMES.items()},
    }
    (project_dir / "garbage.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(
        f"train={len(train_names)}, val={len(val_names)}, "
        f"objects={count_objects}, empty frames={count_empty}"
    )


def main():
    """Читаем пути и обрабатываем датасет для модели."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset(args.project.resolve(), args.images.resolve())


if __name__ == "__main__":
    main()
