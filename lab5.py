from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "lab5_data"
data_folder.mkdir(exist_ok=True)

input_path = data_folder / "dataIn.txt"
output_path = data_folder / "dataOut.txt"

# Создаём демонстрационные данные только при отсутствии обоих файлов.
# Существующие данные при повторном запуске не перезаписываем.
if not input_path.exists() and not output_path.exists():
    rng = np.random.RandomState(42)

    # 100 примеров, каждый содержит 12 бинарных ответов.
    generated_X = rng.randint(0, 2, size=(100, 12))

    # Сначала генерируем одну метку на пример:
    # 0 — правящая партия, 1 — оппозиция.
    generated_labels = rng.randint(0, 2, size=100)

    # Корректные ответы: только [1, 0] или [0, 1].
    generated_Y = np.eye(2, dtype=int)[generated_labels]

    # По описанию методички примеры в файлах расположены столбцами:
    # входы 12 × N, ответы 2 × N.
    np.savetxt(input_path, generated_X.T, fmt="%d")
    np.savetxt(output_path, generated_Y.T, fmt="%d")

    print("Созданы синтетические данные со случайными классами.")

elif not input_path.exists() or not output_path.exists():
    raise FileNotFoundError(
        "Для загрузки нужны оба файла: dataIn.txt и dataOut.txt."
    )

# Транспонируем: для вычислений каждый пример располагаем строкой.
X = np.loadtxt(input_path, ndmin=2).T
Y = np.loadtxt(output_path, ndmin=2).T

# Проверяем размеры и допустимость значений.
if X.shape[1] != 12 or Y.shape[1] != 2:
    raise ValueError("Ожидаются файлы размеров 12 × N и 2 × N.")

if X.shape[0] != Y.shape[0]:
    raise ValueError("Количество входных примеров и ответов различается.")

if not np.isin(X, [0, 1]).all():
    raise ValueError("Входные ответы должны содержать только 0 и 1.")

if not np.isin(Y, [0, 1]).all() or not np.all(Y.sum(axis=1) == 1):
    raise ValueError("Допустимые выходы: только [1, 0] и [0, 1].")

# Преобразуем двухкомпонентные ответы в номера классов.
labels = Y.argmax(axis=1)

# Отделяем 30% примеров для итогового тестирования.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# Из оставшихся 70 примеров выделяем 20 для валидации.
# Итоговое разбиение: 50 — обучение, 20 — валидация, 30 — тест.
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=20,
    random_state=42,
    stratify=y_train
)

print("\nВсего примеров:", X.shape[0])
print("Обучение:", X_train.shape)
print("Валидация:", X_val.shape)
print("Тест:", X_test.shape)

print("\nКоличество примеров каждого класса:")
print("Обучение:", np.bincount(y_train, minlength=2))
print("Валидация:", np.bincount(y_val, minlength=2))
print("Тест:", np.bincount(y_test, minlength=2))

print("\nПервые пять входных примеров:")
print(X[:5].astype(int))

print("\nПервые пять правильных ответов:")
print(Y[:5].astype(int))


# Архитектура: 12 входов → 8 скрытых нейронов → 2 выхода.
rng = np.random.RandomState(42)

# Начальные веса выбираем случайно, смещения задаём нулями.
W1 = rng.normal(0, np.sqrt(1 / 12), size=(12, 8))
b1 = np.zeros((1, 8))

W2 = rng.normal(0, np.sqrt(1 / 8), size=(8, 2))
b2 = np.zeros((1, 2))


def sigmoid(z):
    # Устойчивая реализация logsig без переполнения экспоненты.
    return np.exp(-np.logaddexp(0, -z))


def forward(inputs):
    # Прямой проход через скрытый слой.
    hidden = sigmoid(inputs @ W1 + b1)

    # Выходной слой: Softmax.
    logits = hidden @ W2 + b2
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return hidden, probabilities


def cross_entropy(true_labels, probabilities):
    # Ошибка по вероятности правильного класса.
    correct_probabilities = probabilities[
        np.arange(len(true_labels)), true_labels
    ]
    return -np.mean(np.log(np.clip(correct_probabilities, 1e-15, 1)))


learning_rate = 0.1
epochs = 2000

# Правильные ответы обучения в двухкомпонентном формате.
Y_train = np.eye(2)[y_train]

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_epoch = 0
best_parameters = None

for epoch in range(1, epochs + 1):
    hidden, probabilities = forward(X_train)

    # Обратное распространение ошибки.
    # Для Softmax с кросс-энтропией градиент по logits:
    output_gradient = (probabilities - Y_train) / len(X_train)

    dW2 = hidden.T @ output_gradient
    db2 = output_gradient.sum(axis=0, keepdims=True)

    hidden_gradient = (
        output_gradient @ W2.T
    ) * hidden * (1 - hidden)

    dW1 = X_train.T @ hidden_gradient
    db1 = hidden_gradient.sum(axis=0, keepdims=True)

    # Все градиенты вычислены до изменения весов.
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Измеряем ошибки после обновления параметров.
    _, train_probabilities = forward(X_train)
    _, val_probabilities = forward(X_val)

    train_loss = cross_entropy(y_train, train_probabilities)
    val_loss = cross_entropy(y_val, val_probabilities)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Выбираем состояние сети по валидации, не используя тест.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_parameters = (
            W1.copy(), b1.copy(),
            W2.copy(), b2.copy()
        )

    if epoch == 1 or epoch % 200 == 0:
        print(
            f"Эпоха {epoch:4d}: "
            f"ошибка обучения = {train_loss:.4f}, "
            f"ошибка валидации = {val_loss:.4f}"
        )

# Восстанавливаем параметры с минимальной ошибкой валидации.
W1, b1, W2, b2 = best_parameters

print("\nВыбрана эпоха:", best_epoch)
print(f"Минимальная ошибка валидации: {best_val_loss:.4f}")

# Итоговую тестовую оценку считаем после выбора состояния сети.
from sklearn.metrics import accuracy_score, classification_report

_, train_probabilities = forward(X_train)
_, test_probabilities = forward(X_test)

train_predictions = train_probabilities.argmax(axis=1)
test_predictions = test_probabilities.argmax(axis=1)

print(f"\nAccuracy на обучении: "
      f"{accuracy_score(y_train, train_predictions):.4f}")
print(f"Accuracy на тесте: "
      f"{accuracy_score(y_test, test_predictions):.4f}")

# Ориентир: всегда предсказывать наиболее частый обучающий класс.
majority_class = np.bincount(y_train, minlength=2).argmax()
baseline_predictions = np.full(len(y_test), majority_class)

print(f"Accuracy постоянного прогноза: "
      f"{accuracy_score(y_test, baseline_predictions):.4f}")

print("\nОтчёт по классификации:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Правящая партия", "Оппозиция"],
    digits=4,
    zero_division=0
))


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

results_folder = project_folder / "lab5_results"
results_folder.mkdir(exist_ok=True)

# История обучения относится ко всем 2000 эпохам.
# Итоговые прогнозы получены восстановленной моделью 25-й эпохи.
history_table = pd.DataFrame({
    "epoch": np.arange(1, epochs + 1),
    "train_loss": train_losses,
    "validation_loss": val_losses
})
history_table.to_csv(
    results_folder / "training_history.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history_table["epoch"], train_losses, label="Обучение")
ax.plot(history_table["epoch"], val_losses, label="Валидация")
ax.axvline(
    best_epoch,
    color="gray",
    linestyle="--",
    label=f"Выбранная эпоха: {best_epoch}"
)
ax.set(
    xlabel="Эпоха",
    ylabel="Кросс-энтропия",
    title="Изменение ошибки при обучении MLP"
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(results_folder / "training_loss.png", dpi=200)
plt.close(fig)

# Матрица ошибок итоговой модели.
cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Правящая партия", "Оппозиция"]
)
display.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set(
    xlabel="Предсказанный класс",
    ylabel="Настоящий класс",
    title="Матрица ошибок MLP на тесте"
)
fig.tight_layout()
fig.savefig(results_folder / "confusion_matrix.png", dpi=200)
plt.close(fig)

# Показываем вероятность оппозиции для каждого тестового примера.
fig, ax = plt.subplots(figsize=(9, 5))
example_numbers = np.arange(1, len(y_test) + 1)

ax.plot(
    example_numbers,
    test_probabilities[:, 1],
    "o",
    label="Вероятность класса «Оппозиция»"
)
ax.scatter(
    example_numbers,
    y_test,
    marker="x",
    color="black",
    label="Настоящий класс: 0 или 1"
)
ax.axhline(0.5, color="gray", linestyle="--", label="Порог 0,5")
ax.set(
    xlabel="Номер тестового примера",
    ylabel="Вероятность / метка класса",
    title="Прогнозы MLP на синтетических данных",
    ylim=(-0.05, 1.05)
)
ax.legend()
fig.tight_layout()
fig.savefig(results_folder / "predictions.png", dpi=200)
plt.close(fig)

# Сохраняем входные ответы, метки и вероятности тестовых примеров.
predictions_table = pd.DataFrame(
    X_test.astype(int),
    columns=[f"question_{i}" for i in range(1, 13)]
)
predictions_table["actual_class"] = y_test
predictions_table["predicted_class"] = test_predictions
predictions_table["probability_ruling"] = test_probabilities[:, 0]
predictions_table["probability_opposition"] = test_probabilities[:, 1]

predictions_table.to_csv(
    results_folder / "test_predictions.csv",
    index=False
)

# Сохраняем параметры выбранной модели.
np.savez(
    results_folder / "model_weights.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

report = (
    "Лабораторная работа №5 — многослойный перцептрон\n\n"
    "Данные: 100 синтетических примеров со случайными классами.\n"
    "Классы сгенерированы независимо от входных ответов.\n"
    "0 — правящая партия; 1 — оппозиция.\n"
    "Разбиение: 50 обучение, 20 валидация, 30 тест.\n"
    "Архитектура: 12 → 8 → 2.\n"
    "Скрытая активация: logsig; выходная: Softmax.\n"
    "Оптимизация: полный градиентный спуск.\n"
    f"Скорость обучения: {learning_rate}\n"
    f"Количество эпох: {epochs}\n"
    f"Выбранная по валидации эпоха: {best_epoch}\n"
    f"Минимальная ошибка валидации: {best_val_loss:.4f}\n\n"
    f"Accuracy на обучении: "
    f"{accuracy_score(y_train, train_predictions):.4f}\n"
    f"Accuracy на тесте: "
    f"{accuracy_score(y_test, test_predictions):.4f}\n"
    f"Accuracy постоянного прогноза: "
    f"{accuracy_score(y_test, baseline_predictions):.4f}\n\n"
    + str(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Правящая партия", "Оппозиция"],
        digits=4,
        zero_division=0
    ))
    + "\nРезультат не характеризует качество реального "
      "прогнозирования выборов.\n"
)

(results_folder / "metrics.txt").write_text(
    report,
    encoding="utf-8"
)

print("\nРезультаты сохранены в:", results_folder)