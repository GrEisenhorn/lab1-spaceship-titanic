from pathlib import Path
import pandas as pd

# Загружаем подготовленные в первой работе выборки.
project_folder = Path(__file__).resolve().parent

train_df = pd.read_csv(project_folder / "train_processed.csv")
test_df = pd.read_csv(project_folder / "test_processed.csv")

print("Размер обучающей выборки:", train_df.shape)
print("Размер тестовой выборки:", test_df.shape)

print("\nПропуски в обучающей выборке:", train_df.isnull().sum().sum())
print("Пропуски в тестовой выборке:", test_df.isnull().sum().sum())

# Для регрессии выбираем возраст Age как целевую переменную y.
# Из входных признаков X исключаем сам возраст и Transported:
# результат переноса пассажира не используем для прогнозирования возраста.
X_train_reg = train_df.drop(columns=["Age", "Transported"])
X_test_reg = test_df.drop(columns=["Age", "Transported"])

y_train_reg = train_df["Age"]
y_test_reg = test_df["Age"]

print("\nПодготовка к прогнозированию возраста:")
print("Обучающие признаки:", X_train_reg.shape)
print("Тестовые признаки:", X_test_reg.shape)
print("Первые пять значений целевой переменной:")
print(y_train_reg.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Восстанавливаем исходные возрасты и то же разбиение,
# которое использовалось в первой лабораторной.
original_df = pd.read_csv(project_folder / "train.csv")

original_train, original_test = train_test_split(
    original_df,
    test_size=0.3,
    random_state=42,
    stratify=original_df["Transported"]
)

# Порядок строк соответствует обработанным CSV.
# Сбрасываем индексы, чтобы корректно выбирать те же строки.
original_train = original_train.reset_index(drop=True)
original_test = original_test.reset_index(drop=True)

# Исключаем строки, где исходный возраст отсутствовал:
# подставленная медиана не является известным правильным ответом.
train_age_known = original_train["Age"].notna()
test_age_known = original_test["Age"].notna()

X_train_reg = X_train_reg.loc[train_age_known].copy()
X_test_reg = X_test_reg.loc[test_age_known].copy()

# Целевая переменная теперь содержит возраст в годах.
y_train_reg = original_train.loc[train_age_known, "Age"]
y_test_reg = original_test.loc[test_age_known, "Age"]

print("\nСтрок с известным возрастом:")
print("Обучающая выборка:", len(y_train_reg))
print("Тестовая выборка:", len(y_test_reg))

# Обучаем линейную регрессию на обучающей выборке.
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# Получаем прогнозы для обучения и независимой тестовой выборки.
train_predictions = reg_model.predict(X_train_reg)
test_predictions = reg_model.predict(X_test_reg)

# MAE — средняя абсолютная ошибка в годах.
# RMSE сильнее учитывает большие ошибки, также выражается в годах.
# R² характеризует качество относительно постоянного прогноза:
# 1 — идеальный результат; значение может быть отрицательным.
print("\nОценка линейной регрессии:")
print(f"MAE на обучении: "
      f"{mean_absolute_error(y_train_reg, train_predictions):.2f} года")
print(f"MAE на тесте: "
      f"{mean_absolute_error(y_test_reg, test_predictions):.2f} года")
print(f"RMSE на тесте: "
      f"{mean_squared_error(y_test_reg, test_predictions) ** 0.5:.2f} года")
print(f"R² на тесте: {r2_score(y_test_reg, test_predictions):.4f}")

# Сравниваем модель с простым прогнозом:
# всем пассажирам назначается средний возраст обучающей выборки.
baseline_predictions = [y_train_reg.mean()] * len(y_test_reg)

print(f"\nMAE прогноза средним возрастом: "
      f"{mean_absolute_error(y_test_reg, baseline_predictions):.2f} года")

comparison = pd.DataFrame({
    "Настоящий возраст": y_test_reg,
    "Прогноз модели": test_predictions
})

print("\nПримеры прогнозов:")
print(comparison.head(10).round(1).to_string(index=False))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Для классификации используем все строки таблиц.
# Исключённые ранее неизвестные возрасты касались только регрессии.
# Transported убираем из признаков и используем как правильный ответ.
X_train_cls = train_df.drop(columns=["Transported"])
X_test_cls = test_df.drop(columns=["Transported"])

y_train_cls = train_df["Transported"]
y_test_cls = test_df["Transported"]

# Обучаем логистическую регрессию.
# max_iter задаёт максимальное число итераций оптимизации.
cls_model = LogisticRegression(max_iter=2000)
cls_model.fit(X_train_cls, y_train_cls)

train_predictions_cls = cls_model.predict(X_train_cls)
test_predictions_cls = cls_model.predict(X_test_cls)

print("\nОценка логистической регрессии:")
print(f"Accuracy на обучении: "
      f"{accuracy_score(y_train_cls, train_predictions_cls):.4f}")
print(f"Accuracy на тесте: "
      f"{accuracy_score(y_test_cls, test_predictions_cls):.4f}")

# Сравниваем с постоянным прогнозом наиболее частого класса.
majority_class = y_train_cls.mode().iloc[0]
baseline_cls = [majority_class] * len(y_test_cls)

print(f"Accuracy прогноза наиболее частым классом: "
      f"{accuracy_score(y_test_cls, baseline_cls):.4f}")

# Отчёт содержит precision, recall и F1 отдельно для каждого класса.
print("\nОтчёт по классификации:")
print(classification_report(
    y_test_cls,
    test_predictions_cls,
    labels=[0, 1],
    target_names=["Не перенесён", "Перенесён"],
    digits=4,
    zero_division=0
))

# Строки — настоящие классы, столбцы — предсказанные.
print("Матрица ошибок:")
print(confusion_matrix(
    y_test_cls,
    test_predictions_cls,
    labels=[0, 1]
))


import matplotlib.pyplot as plt
import seaborn as sns

# Все результаты работы сохраняем в отдельную папку.
results_folder = project_folder / "lab2_results"
results_folder.mkdir(exist_ok=True)

# График качества регрессии.
# На диагонали находятся точки с идеально предсказанным возрастом.
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(y_test_reg, test_predictions, alpha=0.25, s=15)

lower = min(y_test_reg.min(), test_predictions.min())
upper = max(y_test_reg.max(), test_predictions.max())
ax.plot([lower, upper], [lower, upper], "r--", label="Идеальный прогноз")

ax.set(
    xlabel="Настоящий возраст, лет",
    ylabel="Предсказанный возраст, лет",
    title="Линейная регрессия — прогноз возраста"
)
ax.legend()
fig.tight_layout()
fig.savefig(results_folder / "regression.png", dpi=200)
plt.close(fig)

# Визуализация матрицы ошибок классификации.
cm = confusion_matrix(y_test_cls, test_predictions_cls, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Не перенесён", "Перенесён"],
    yticklabels=["Не перенесён", "Перенесён"],
    ax=ax
)
ax.set(
    xlabel="Предсказанный класс",
    ylabel="Настоящий класс",
    title="Матрица ошибок логистической регрессии"
)
fig.tight_layout()
fig.savefig(results_folder / "confusion_matrix.png", dpi=200)
plt.close(fig)

# Сохраняем прогнозы регрессии и классификации.
comparison.to_csv(
    results_folder / "regression_predictions.csv",
    index=False
)

pd.DataFrame({
    "Настоящий класс": y_test_cls,
    "Предсказанный класс": test_predictions_cls
}).to_csv(
    results_folder / "classification_predictions.csv",
    index=False
)

# Формируем текстовый отчёт с метриками обеих моделей.
report = (
    "Лабораторная работа №2\n\n"
    "Линейная регрессия — прогноз возраста\n"
    f"MAE на обучении: "
    f"{mean_absolute_error(y_train_reg, train_predictions):.4f}\n"
    f"MAE на тесте: "
    f"{mean_absolute_error(y_test_reg, test_predictions):.4f}\n"
    f"RMSE на тесте: "
    f"{mean_squared_error(y_test_reg, test_predictions) ** 0.5:.4f}\n"
    f"R² на тесте: {r2_score(y_test_reg, test_predictions):.4f}\n"
    f"MAE прогноза средним: "
    f"{mean_absolute_error(y_test_reg, baseline_predictions):.4f}\n\n"
    "Логистическая регрессия — прогноз Transported\n"
    f"Accuracy на обучении: "
    f"{accuracy_score(y_train_cls, train_predictions_cls):.4f}\n"
    f"Accuracy на тесте: "
    f"{accuracy_score(y_test_cls, test_predictions_cls):.4f}\n"
    f"Accuracy прогноза наиболее частым классом: "
    f"{accuracy_score(y_test_cls, baseline_cls):.4f}\n\n"
    + str(classification_report(
    y_test_cls,
    test_predictions_cls,
    labels=[0, 1],
    target_names=["Не перенесён", "Перенесён"],
    digits=4,
    zero_division=0
))
    + "\nМатрица ошибок:\n"
    + str(cm)
)

(results_folder / "metrics.txt").write_text(report, encoding="utf-8")

print("\nРезультаты второй лабораторной сохранены в:")
print(results_folder)