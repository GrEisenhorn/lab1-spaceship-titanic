from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

project_folder = Path(__file__).resolve().parent

# Используем подготовленные выборки первой лабораторной.
train_df = pd.read_csv(project_folder / "train_processed.csv")
test_df = pd.read_csv(project_folder / "test_processed.csv")

# Исходные данные нужны для восстановления возраста в годах
# и исключения строк, где возраст был заполнен медианой.
original_df = pd.read_csv(project_folder / "train.csv")

original_train, original_test = train_test_split(
    original_df,
    test_size=0.3,
    random_state=42,
    stratify=original_df["Transported"]
)

original_train = original_train.reset_index(drop=True)
original_test = original_test.reset_index(drop=True)

# Проверяем соответствие возраста и порядка строк обработанным CSV.
# В первой работе возраст масштабировался по обучающей выборке.
age_min = original_train["Age"].min()
age_max = original_train["Age"].max()
age_median = original_train["Age"].median()

for processed, original in [
    (train_df, original_train),
    (test_df, original_test)
]:
    expected_age = (
        original["Age"].fillna(age_median) - age_min
    ) / (age_max - age_min)

    if not np.allclose(processed["Age"], expected_age):
        raise ValueError(
            "Возраст или порядок строк не соответствует первой работе."
        )

# Регрессия: предсказываем только известный исходный возраст.
# Transported не используем как признак для прогнозирования возраста.
train_age_known = original_train["Age"].notna()
test_age_known = original_test["Age"].notna()

X_train_reg = train_df.loc[train_age_known].drop(
    columns=["Age", "Transported"]
)
X_test_reg = test_df.loc[test_age_known].drop(
    columns=["Age", "Transported"]
)

y_train_reg = original_train.loc[train_age_known, "Age"]
y_test_reg = original_test.loc[test_age_known, "Age"]

# Классификация: используем все строки, целевой столбец — Transported.
X_train_cls = train_df.drop(columns=["Transported"])
X_test_cls = test_df.drop(columns=["Transported"])

y_train_cls = train_df["Transported"]
y_test_cls = test_df["Transported"]

print("Данные для регрессии:")
print("Обучение:", X_train_reg.shape)
print("Тест:", X_test_reg.shape)

print("\nДанные для классификации:")
print("Обучение:", X_train_cls.shape)
print("Тест:", X_test_cls.shape)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Начальная модель с заранее заданными ограничениями сложности.
# max_depth — максимальная глубина дерева.
# min_samples_leaf — минимальное число обучающих объектов в листе.
reg_tree = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=30,
    random_state=42
)

reg_tree.fit(X_train_reg, y_train_reg)

# Прогнозы для обучающей и тестовой выборок.
train_predictions_reg = reg_tree.predict(X_train_reg)
test_predictions_reg = reg_tree.predict(X_test_reg)

print("\nДерево решений — прогноз возраста:")

print(f"MAE на обучении: "
      f"{mean_absolute_error(y_train_reg, train_predictions_reg):.2f} года")

print(f"MAE на тесте: "
      f"{mean_absolute_error(y_test_reg, test_predictions_reg):.2f} года")

print(f"RMSE на тесте: "
      f"{mean_squared_error(y_test_reg, test_predictions_reg) ** 0.5:.2f} года")

print(f"R² на обучении: "
      f"{r2_score(y_train_reg, train_predictions_reg):.4f}")

print(f"R² на тесте: "
      f"{r2_score(y_test_reg, test_predictions_reg):.4f}")

# Постоянный прогноз нужен как ориентир качества.
baseline_reg = [y_train_reg.mean()] * len(y_test_reg)

print(f"\nMAE прогноза средним возрастом: "
      f"{mean_absolute_error(y_test_reg, baseline_reg):.2f} года")

comparison_reg = pd.DataFrame({
    "Настоящий возраст": y_test_reg,
    "Прогноз дерева": test_predictions_reg
})

print("\nПримеры прогнозов:")
print(comparison_reg.head(10).round(1).to_string(index=False))


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt

# Начальная модель с ограничениями сложности.
cls_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=30,
    random_state=42
)

cls_tree.fit(X_train_cls, y_train_cls)

train_predictions_cls = cls_tree.predict(X_train_cls)
test_predictions_cls = cls_tree.predict(X_test_cls)

# Для ROC нужны вероятности класса 1, а не готовые метки 0/1.
# Находим столбец класса 1 в результате predict_proba().
positive_column = list(cls_tree.classes_).index(1)
test_probabilities_cls = cls_tree.predict_proba(
    X_test_cls
)[:, positive_column]

print("\nДерево решений — классификация Transported:")
print(f"Accuracy на обучении: "
      f"{accuracy_score(y_train_cls, train_predictions_cls):.4f}")
print(f"Accuracy на тесте: "
      f"{accuracy_score(y_test_cls, test_predictions_cls):.4f}")

print("\nОтчёт по классификации:")
print(classification_report(
    y_test_cls,
    test_predictions_cls,
    labels=[0, 1],
    target_names=["Не перенесён", "Перенесён"],
    digits=4,
    zero_division=0
))

# ROC показывает соотношение ложных срабатываний и обнаружений
# класса 1 при разных порогах вероятности.
fpr, tpr, thresholds = roc_curve(
    y_test_cls,
    test_probabilities_cls
)

roc_auc = roc_auc_score(y_test_cls, test_probabilities_cls)
print(f"ROC AUC на тесте: {roc_auc:.4f}")

# Создаём папку и сохраняем ROC-кривую.
results_folder = project_folder / "lab3_results"
results_folder.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(fpr, tpr, label=f"Дерево решений, AUC = {roc_auc:.3f}")
ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Случайное ранжирование, AUC = 0.5"
)

ax.set(
    xlabel="Доля ложных срабатываний (FPR)",
    ylabel="Доля обнаруженных объектов класса 1 (TPR)",
    title="ROC-кривая дерева решений",
    xlim=(0, 1),
    ylim=(0, 1.02)
)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(results_folder / "roc_curve.png", dpi=200)
plt.close(fig)

print("\nROC-кривая сохранена:")
print(results_folder / "roc_curve.png")


# Сохраняем прогнозы регрессии.
comparison_reg.to_csv(
    results_folder / "regression_predictions.csv",
    index=False
)

# Для классификации сохраняем и класс, и вероятность класса 1.
pd.DataFrame({
    "Настоящий класс": y_test_cls,
    "Предсказанный класс": test_predictions_cls,
    "Вероятность переноса": test_probabilities_cls
}).to_csv(
    results_folder / "classification_predictions.csv",
    index=False
)

# Сравнение настоящего возраста с прогнозом дерева.
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test_reg, test_predictions_reg, alpha=0.25, s=15)

lower = min(y_test_reg.min(), test_predictions_reg.min())
upper = max(y_test_reg.max(), test_predictions_reg.max())
ax.plot([lower, upper], [lower, upper], "r--", label="Идеальный прогноз")

ax.set(
    xlabel="Настоящий возраст, лет",
    ylabel="Предсказанный возраст, лет",
    title="Дерево решений — прогноз возраста"
)
ax.legend()
fig.tight_layout()
fig.savefig(results_folder / "regression.png", dpi=200)
plt.close(fig)

# Текстовый отчёт с параметрами и оценками моделей.
report = (
    "Лабораторная работа №3\n"
    "Деревья решений\n"
    "Параметры обеих моделей: max_depth=5, min_samples_leaf=30\n"
    "Параметры заданы заранее; автоматический подбор не проводился.\n\n"
    "Регрессия возраста — ошибки в годах\n"
    f"MAE на обучении: "
    f"{mean_absolute_error(y_train_reg, train_predictions_reg):.4f}\n"
    f"MAE на тесте: "
    f"{mean_absolute_error(y_test_reg, test_predictions_reg):.4f}\n"
    f"RMSE на тесте: "
    f"{mean_squared_error(y_test_reg, test_predictions_reg) ** 0.5:.4f}\n"
    f"R² на тесте: "
    f"{r2_score(y_test_reg, test_predictions_reg):.4f}\n"
    f"MAE прогноза средним: "
    f"{mean_absolute_error(y_test_reg, baseline_reg):.4f}\n\n"
    "Классификация Transported\n"
    f"Accuracy на обучении: "
    f"{accuracy_score(y_train_cls, train_predictions_cls):.4f}\n"
    f"Accuracy на тесте: "
    f"{accuracy_score(y_test_cls, test_predictions_cls):.4f}\n"
    f"ROC AUC на тесте: {roc_auc:.4f}\n\n"
    + str(classification_report(
        y_test_cls,
        test_predictions_cls,
        labels=[0, 1],
        target_names=["Не перенесён", "Перенесён"],
        digits=4,
        zero_division=0
    ))
)

(results_folder / "metrics.txt").write_text(report, encoding="utf-8")

print("\nВсе результаты сохранены в:", results_folder)