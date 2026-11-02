from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "lab6_data"

# Находим папку, внутри которой расположены классы жестов.
candidates = [
    folder.parent
    for folder in data_folder.rglob("call")
    if folder.is_dir() and (folder.parent / "fist").is_dir()
]

if len(candidates) != 1:
    raise RuntimeError(
        f"Найдено папок датасета: {len(candidates)}. "
        "Проверь расположение данных в lab6_data."
    )

dataset_folder = candidates[0]
print("Папка датасета:", dataset_folder)

# Считаем изображения отдельно для каждого класса.
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
total = 0

for class_folder in sorted(dataset_folder.iterdir()):
    if not class_folder.is_dir():
        continue

    count = sum(
        1
        for file in class_folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )

    total += count
    print(f"{class_folder.name}: {count}")

print("\nВсего изображений:", total)


import csv
import random
from collections import Counter

project_folder = Path(__file__).resolve().parent
results_folder = project_folder / "lab6_results_improved"
results_folder.mkdir(exist_ok=True)

rng = random.Random(42)
samples_per_class = 1000
records = []

class_folders = sorted(
    folder
    for folder in dataset_folder.iterdir()
    if folder.is_dir()
)

for class_id, class_folder in enumerate(class_folders):
    files = sorted(
        file
        for file in class_folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )

    if len(files) < samples_per_class:
        raise RuntimeError(
            f"В классе {class_folder.name} недостаточно изображений."
        )

    # Случайно выбираем 1000 изображений без повторений.
    selected = rng.sample(files, samples_per_class)

    # Разделяем каждый класс отдельно, сохраняя баланс выборок.
    for index, file in enumerate(selected):
        if index < 700:
            split = "train"
        elif index < 850:
            split = "validation"
        else:
            split = "test"

        records.append({
            "path": file.relative_to(dataset_folder).as_posix(),
            "class_id": class_id,
            "class_name": class_folder.name,
            "split": split,
        })

# Сохраняем список файлов и их распределение по выборкам.
manifest_path = results_folder / "dataset_split.csv"

with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["path", "class_id", "class_name", "split"]
    )
    writer.writeheader()
    writer.writerows(records)

counts = Counter(record["split"] for record in records)

print("\nПодготовленные выборки:")
print("Обучение:", counts["train"])
print("Валидация:", counts["validation"])
print("Тест:", counts["test"])
print("Список изображений сохранён:", manifest_path)


import json
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

image_size = (64, 64)
batch_size = 32
class_names = [folder.name for folder in class_folders]


def load_image(path, label):
    # Читаем изображение, приводим к RGB и размеру 64 × 64.
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label


def make_dataset(split, training=False):
    selected = [
        record for record in records
        if record["split"] == split
    ]

    paths = [
        str(dataset_folder / record["path"])
        for record in selected
    ]
    labels = [record["class_id"] for record in selected]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Перемешиваем только обучающую выборку.
    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(2)


train_data = make_dataset("train", training=True)
val_data = make_dataset("validation")
test_data = make_dataset("test")

# Архитектура CNN из методички.
model = tf.keras.Sequential([
    tf.keras.Input(shape=(64, 64, 3)),

    # При обучении случайно поворачиваем изображения примерно на ±29°.
    tf.keras.layers.RandomRotation(
        factor=0.08,
        fill_mode="reflect",
        seed=42
    ),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),

    # Во время обучения случайно отключаем 40% выходов этого слоя.
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    ),
])

# Метки представлены номерами классов, поэтому используем sparse-вариант.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_path = results_folder / "best_model.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3
    ),
]

print("\nОбучение CNN...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks,
    shuffle=False
)

# Загружаем модель с минимальной ошибкой валидации.
model = tf.keras.models.load_model(str(model_path))

print("\nОценка лучшей модели на тесте:")
test_loss, test_accuracy = model.evaluate(test_data)

print(f"Ошибка на тесте: {test_loss:.4f}")
print(f"Accuracy на тесте: {test_accuracy:.4f}")

# Сохраняем историю обучения и порядок выходных классов.
with (results_folder / "training_history.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch", "loss", "accuracy", "val_loss", "val_accuracy"
    ])

    for index in range(len(history.history["loss"])):
        writer.writerow([
            index + 1,
            history.history["loss"][index],
            history.history["accuracy"][index],
            history.history["val_loss"][index],
            history.history["val_accuracy"][index],
        ])

(results_folder / "class_names.json").write_text(
    json.dumps(class_names, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nМодель сохранена:", model_path)

