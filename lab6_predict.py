from pathlib import Path
import json
import tensorflow as tf

project_folder = Path(__file__).resolve().parent
results_folder = project_folder / "lab6_results_improved"
photo_path = project_folder / "lab6_photos" / "my_gesture_2.jpg"

if not photo_path.is_file():
    raise FileNotFoundError(f"Фотография не найдена: {photo_path}")

# Загружаем обученную модель и названия её выходных классов.
model = tf.keras.models.load_model(
    str(results_folder / "best_model.keras")
)

class_names = json.loads(
    (results_folder / "class_names.json").read_text(encoding="utf-8")
)

# Обрабатываем фотографию так же, как изображения при обучении.
image_bytes = tf.io.read_file(str(photo_path))
image = tf.io.decode_image(
    image_bytes,
    channels=3,
    expand_animations=False
)
image = tf.image.resize(image, (64, 64)) / 255.0

# Добавляем размерность пакета: одно изображение.
probabilities = model.predict(
    tf.expand_dims(image, axis=0),
    verbose=0
)[0]

# Выводим три класса с наибольшими оценками модели.
top_indices = sorted(
    range(len(probabilities)),
    key=lambda index: float(probabilities[index]),
    reverse=True
)[:3]

lines = [
    f"Фотография: {photo_path.name}",
    f"Предсказанный класс: {class_names[top_indices[0]]}",
    "",
    "Три наиболее вероятных класса:"
]

for index in top_indices:
    lines.append(
        f"{class_names[index]}: {float(probabilities[index]):.2%}"
    )

report = "\n".join(lines)
print(report)

(results_folder / f"{photo_path.stem}_prediction.txt").write_text(
    report,
    encoding="utf-8"
)
