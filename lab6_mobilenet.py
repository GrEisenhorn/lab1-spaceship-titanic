import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "lab6_data" / "hagrid-classification-512p-no-gesture-150k"
SOURCE_RESULTS = PROJECT / "lab6_results"
RESULTS = PROJECT / "lab6_results_mobilenet"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def read_image(path):
    """RGB, 128 × 128. Пиксели остаются в диапазоне 0–255."""
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    return tf.image.resize(image, IMAGE_SIZE)


def image_dataset(rows):
    """Порядок изображений совпадает с порядком строк CSV."""
    paths = [str(DATASET / relative_path) for relative_path in rows["path"]]
    return (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(2)
    )


def plot_history(history):
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in zip(
        axes, ["loss", "accuracy"], ["Кросс-энтропия", "Доля верных ответов"]
    ):
        ax.plot(epochs, history.history[metric], label="Обучение")
        ax.plot(epochs, history.history[f"val_{metric}"], label="Валидация")
        ax.axvline(best_epoch, color="gray", linestyle="--", label="Лучшая эпоха")
        ax.set(xlabel="Эпоха", ylabel=label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("MobileNetV2: обучение классификатора на фиксированных признаках")
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=180)
    plt.close(fig)


def predict_photos(model, class_names):
    """Обрабатываем все фотографии в lab6_photos тем же способом, что и тест."""
    folder = PROJECT / "lab6_photos"
    photos = sorted(
        path for path in folder.glob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not photos:
        print("Фотографии не найдены в lab6_photos. Проверку можно выполнить позже.")
        return

    reports = []
    for path in photos:
        image = read_image(str(path))
        probabilities = model.predict(image[None, ...], verbose=0)[0]
        top = np.argsort(probabilities)[::-1][:3]
        lines = [f"Фотография: {path.name}", "Три наибольшие оценки модели:"]
        for index in top:
            lines.append(f"{class_names[index]}: {probabilities[index]:.2%}")
        lines.append("Оценки softmax не гарантируют правильного распознавания.")
        report = "\n".join(lines)
        print("\n" + report)
        (RESULTS / f"{path.stem}_prediction.txt").write_text(report, encoding="utf-8")
        reports.append({"photo": path.name, "predicted_class": class_names[top[0]],
                        "score": float(probabilities[top[0]])})

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.clip(image.numpy() / 255.0, 0, 1))
        ax.set_title(f"{path.name}\n{class_names[top[0]]}: {probabilities[top[0]]:.1%}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(RESULTS / f"{path.stem}_prediction.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(reports).to_csv(RESULTS / "photo_predictions.csv", index=False, encoding="utf-8-sig")


def main():
    tf.keras.utils.set_random_seed(SEED)
    manifest = SOURCE_RESULTS / "dataset_split.csv"
    labels_path = SOURCE_RESULTS / "class_names.json"
    for path in [DATASET, manifest, labels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Не найдено: {path}")

    # Используем готовое разделение, не выбираем новые случайные изображения.
    table = pd.read_csv(manifest, encoding="utf-8-sig")
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    required = {"path", "class_id", "class_name", "split"}
    if not required.issubset(table.columns):
        raise ValueError("В dataset_split.csv отсутствуют необходимые столбцы.")
    if table["path"].duplicated().any():
        raise ValueError("В разделении есть повторяющиеся пути к изображениям.")
    for row in table.itertuples():
        class_id = int(row.class_id)
        if not (0 <= class_id < len(class_names)) or class_names[class_id] != row.class_name:
            raise ValueError("Номера классов CSV не совпадают с class_names.json.")
        if not (DATASET / row.path).is_file():
            raise FileNotFoundError(f"Не найдено изображение: {DATASET / row.path}")
    if set(table["split"]) != {"train", "validation", "test"}:
        raise ValueError("Ожидаются выборки train, validation и test.")

    RESULTS.mkdir(exist_ok=True)
    shutil.copy2(manifest, RESULTS / manifest.name)
    (RESULTS / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = {split: table[table["split"] == split].reset_index(drop=True)
            for split in ["train", "validation", "test"]}
    for split, subset in rows.items():
        print(f"{split}: {len(subset)} изображений")

    print("\nЗагрузка MobileNetV2. При первом запуске нужен интернет для скачивания весов.")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False

    # Нормализация MobileNetV2: из 0–255 в -1–1. Она включена в саму модель.
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_0_255")
    scaled = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    features = base(scaled, training=False)
    extractor = tf.keras.Model(inputs, features, name="frozen_mobilenet")

    print("\nИзвлечение признаков обучения. Это может занять несколько минут.")
    train_features = extractor.predict(image_dataset(rows["train"]), verbose=1)
    print("\nИзвлечение признаков валидации...")
    val_features = extractor.predict(image_dataset(rows["validation"]), verbose=1)

    # Учим только новый классификатор. Аугментация здесь не используется:
    # признаки уже вычислены и не меняются между эпохами.
    head = tf.keras.Sequential([
        tf.keras.Input(shape=(train_features.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ], name="gesture_classifier")
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    best_head = RESULTS / "best_head.weights.h5"
    print("\nОбучение классификатора...")
    history = head.fit(
        train_features, rows["train"]["class_id"].to_numpy(dtype=np.int32),
        validation_data=(val_features, rows["validation"]["class_id"].to_numpy(dtype=np.int32)),
        epochs=25, batch_size=32, shuffle=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(best_head), monitor="val_loss",
                mode="min", save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=4),
        ],
    )
    head.load_weights(str(best_head))
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nЛучшая эпоха: {best_epoch}")
    print(f"Минимальная ошибка валидации: {min(history.history['val_loss']):.4f}")

    history_table = pd.DataFrame(history.history)
    history_table.insert(0, "epoch", np.arange(1, len(history_table) + 1))
    history_table.to_csv(RESULTS / "training_history.csv", index=False)
    plot_history(history)

    # Итоговый файл принимает изображения 128 × 128 с пикселями 0–255.
    model = tf.keras.Model(inputs, head(features), name="mobilenet_gestures")
    model.save(str(RESULTS / "best_model.keras"))
    print("\nОценка на тесте...")
    probabilities = model.predict(image_dataset(rows["test"]), verbose=1)
    predictions = probabilities.argmax(axis=1)
    y_test = rows["test"]["class_id"].to_numpy(dtype=np.int32)
    accuracy = accuracy_score(y_test, predictions)
    loss = float(tf.keras.losses.sparse_categorical_crossentropy(y_test, probabilities).numpy().mean())
    report = classification_report(y_test, predictions, labels=list(range(len(class_names))),
                                   target_names=class_names, digits=4, zero_division=0)
    train_counts = rows["train"]["class_id"].value_counts()
    majority_class = int(train_counts.idxmax())
    baseline = float(np.mean(y_test == majority_class))
    metrics = (f"MobileNetV2, замороженная база ImageNet, вход 128 × 128\n"
               f"Лучшая эпоха: {best_epoch}\n"
               f"Минимальная ошибка валидации: {min(history.history['val_loss']):.4f}\n"
               f"Accuracy на тесте: {accuracy:.4f}\nОшибка на тесте: {loss:.4f}\n"
               f"Accuracy постоянного прогноза: {baseline:.4f}\n\n{report}\n"
               "Разделение по изображениям не гарантирует разделения по людям.\n")
    print(metrics)
    (RESULTS / "metrics.txt").write_text(metrics, encoding="utf-8")
    output = rows["test"].copy()
    output["predicted_class"] = [class_names[index] for index in predictions]
    for index, name in enumerate(class_names):
        output[f"probability_{name}"] = probabilities[:, index]
    output.to_csv(RESULTS / "test_predictions.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(13, 12))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=list(range(len(class_names))), display_labels=class_names,
        cmap="Blues", xticks_rotation=90, values_format="d", colorbar=False, ax=ax
    )
    ax.set(title="MobileNetV2: матрица ошибок на тесте",
           xlabel="Предсказанный класс", ylabel="Настоящий класс")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    predict_photos(model, class_names)
    print("\nРезультаты сохранены:", RESULTS)


if __name__ == "__main__":
    main()
