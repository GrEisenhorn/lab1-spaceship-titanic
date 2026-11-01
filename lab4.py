from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

# Используем готовое разбиение из первой лабораторной.
project_folder = Path(__file__).resolve().parent

train_df = pd.read_csv(project_folder / "train_processed.csv")
test_df = pd.read_csv(project_folder / "test_processed.csv")

# Transported — целевой столбец, исключаем его из входных признаков.
X_train = train_df.drop(columns=["Transported"])
X_test = test_df.drop(columns=["Transported"])

y_train = train_df["Transported"]
y_test = test_df["Transported"]

print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)

# Параметры заданы заранее, без подбора по тестовой выборке.
# bootstrap=True включает выборки с повторениями для отдельных деревьев.
# oob_score=True включает оценку по объектам, не попавшим в эти выборки.
forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

print("\nОбучение случайного леса...")
forest_model.fit(X_train, y_train)

forest_train_predictions = forest_model.predict(X_train)
forest_test_predictions = forest_model.predict(X_test)

# Для ROC AUC используем вероятности класса 1.
positive_column = list(forest_model.classes_).index(1)
forest_test_probabilities = forest_model.predict_proba(
    X_test
)[:, positive_column]

print("\nРезультаты случайного леса:")

print(f"Accuracy на обучении: "
      f"{accuracy_score(y_train, forest_train_predictions):.4f}")

# По умолчанию oob_score_ для классификации содержит accuracy.
print(f"OOB accuracy: {forest_model.oob_score_:.4f}")
print(f"OOB error: {1 - forest_model.oob_score_:.4f}")

print(f"Accuracy на тесте: "
      f"{accuracy_score(y_test, forest_test_predictions):.4f}")

print(f"ROC AUC на тесте: "
      f"{roc_auc_score(y_test, forest_test_probabilities):.4f}")

# Сравнение с постоянным прогнозом наиболее частого класса.
majority_class = y_train.mode().iloc[0]
baseline_predictions = [majority_class] * len(y_test)

print(f"Accuracy постоянного прогноза: "
      f"{accuracy_score(y_test, baseline_predictions):.4f}")

print("\nОтчёт по классификации:")
print(classification_report(
    y_test,
    forest_test_predictions,
    labels=[0, 1],
    target_names=["Не перенесён", "Перенесён"],
    digits=4,
    zero_division=0
))


from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# AdaBoost объединяет неглубокие деревья.
# learning_rate регулирует вклад каждого дерева в ансамбль.
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

# Градиентный бустинг последовательно добавляет деревья,
# уменьшающие ошибки текущего ансамбля.
gradient_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)

# Параметры заданы заранее, без подбора по тесту.
boosting_models = {
    "AdaBoost": ada_model,
    "Gradient Boosting": gradient_model
}

# Собираем метрики всех трёх моделей для сравнения.
metrics_rows = [{
    "Модель": "Random Forest",
    "Accuracy обучения": accuracy_score(
        y_train, forest_train_predictions
    ),
    "Accuracy теста": accuracy_score(
        y_test, forest_test_predictions
    ),
    "ROC AUC": roc_auc_score(
        y_test, forest_test_probabilities
    )
}]

# Сохраняем прогнозы и вероятности для последующего построения ROC.
test_predictions = {
    "Random Forest": forest_test_predictions
}
test_probabilities = {
    "Random Forest": forest_test_probabilities
}

for name, model in boosting_models.items():
    print(f"\nОбучение {name}...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    positive_column = list(model.classes_).index(1)
    test_proba = model.predict_proba(X_test)[:, positive_column]

    test_predictions[name] = test_pred
    test_probabilities[name] = test_proba

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    metrics_rows.append({
        "Модель": name,
        "Accuracy обучения": train_accuracy,
        "Accuracy теста": test_accuracy,
        "ROC AUC": test_auc
    })

    print(f"Accuracy на обучении: {train_accuracy:.4f}")
    print(f"Accuracy на тесте: {test_accuracy:.4f}")
    print(f"ROC AUC на тесте: {test_auc:.4f}")

    print(classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Не перенесён", "Перенесён"],
        digits=4,
        zero_division=0
    ))

metrics_table = pd.DataFrame(metrics_rows)

print("\nСравнение трёх моделей:")
print(metrics_table.round(4).to_string(index=False))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Создаём отдельную папку результатов четвёртой работы.
results_folder = project_folder / "lab4_results"
results_folder.mkdir(exist_ok=True)

# Строим ROC-кривые по вероятностям класса 1.
fig, ax = plt.subplots(figsize=(8, 6))

for name, probabilities in test_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_value = roc_auc_score(y_test, probabilities)

    ax.plot(
        fpr,
        tpr,
        label=f"{name}, AUC = {auc_value:.4f}"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Случайное ранжирование"
)

ax.set(
    xlabel="Доля ложных срабатываний (FPR)",
    ylabel="Доля обнаруженных объектов класса 1 (TPR)",
    title="Сравнение ROC-кривых ансамблевых моделей",
    xlim=(0, 1),
    ylim=(0, 1.02)
)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(results_folder / "roc_curves.png", dpi=200)
plt.close(fig)

# Сохраняем метрики без округления исходных значений.
metrics_table.to_csv(
    results_folder / "model_comparison.csv",
    index=False
)

# Сохраняем классы и вероятности всех моделей в одной таблице.
predictions_table = pd.DataFrame({
    "Actual_Transported": y_test
})

for name in test_predictions:
    column_name = name.replace(" ", "_")
    predictions_table[f"{column_name}_prediction"] = (
        test_predictions[name]
    )
    predictions_table[f"{column_name}_probability"] = (
        test_probabilities[name]
    )

predictions_table.to_csv(
    results_folder / "predictions.csv",
    index=False
)

# Собираем текстовый отчёт с OOB-оценкой и метриками.
report_parts = [
    "Лабораторная работа №4 — ансамблевые методы",
    "Параметры моделей заданы заранее, без подбора по тесту.",
    "",
    f"Random Forest OOB accuracy: {forest_model.oob_score_:.4f}",
    f"Random Forest OOB error: {1 - forest_model.oob_score_:.4f}",
    "OOB рассчитана на заранее подготовленных признаках.",
    "",
    metrics_table.round(4).to_string(index=False)
]

# Записываем параметры, чтобы результаты можно было воспроизвести.
all_models = {
    "Random Forest": forest_model,
    **boosting_models
}

for name, model in all_models.items():
    report_parts.extend([
        "",
        name,
        "Параметры модели:",
        str(model.get_params(deep=False)),
        "Отчёт на тестовой выборке:",
        str(classification_report(
            y_test,
            test_predictions[name],
            labels=[0, 1],
            target_names=["Не перенесён", "Перенесён"],
            digits=4,
            zero_division=0
        ))
    ])

report = "\n".join(report_parts)

(results_folder / "metrics.txt").write_text(
    report,
    encoding="utf-8"
)

print("\nРезультаты сохранены в:", results_folder)
