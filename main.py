from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

# Загрузка исходных данных Spaceship Titanic.
file_path = Path(__file__).resolve().parent / "train.csv"

df = pd.read_csv(file_path)

# Первичный просмотр данных и подсчёт пропусков.
print("Первые пять строк датасета:")
print(df.head().to_string())

print("\nКоличество строк и столбцов:")
print(df.shape)

print("\nКоличество пропущенных значений в исходной таблице:")
print(df.isnull().sum())

# Исключаем имя и идентификатор: в этой работе признаки из них не извлекаются.
data = df.drop(columns=["Name", "PassengerId"])

# Разбиение 70/30 с сохранением долей целевого класса Transported.
# Статистики для предобработки далее вычисляются только по обучающей выборке.
train_df, test_df = train_test_split(
    data,
    test_size=0.3,
    random_state=42,
    stratify=data["Transported"]
)

train_df = train_df.copy()
test_df = test_df.copy()
print("\nПропуски ДО заполнения — обучающая выборка:")
print(train_df.isnull().sum())
print("\nПропуски ДО заполнения — тестовая выборка:")
print(test_df.isnull().sum())

# Числовые и категориальные признаки; целевой столбец в эти списки не входит.
numeric_columns = [
    "Age", "RoomService", "FoodCourt",
    "ShoppingMall", "Spa", "VRDeck"
]
categorical_columns = [
    "HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"
]

# Числовые пропуски заполняем медианами, устойчивыми к выбросам.
# В тестовой выборке используем медианы обучающей.
for column in numeric_columns:
    median_value = train_df[column].median()

    train_df[column] = train_df[column].fillna(median_value)

    test_df[column] = test_df[column].fillna(median_value)

    print(f"Числовой столбец {column}: медиана = {median_value}")

# Категориальные пропуски заполняем модами обучающей выборки.
# Для Cabin это приближённое заполнение наиболее частой каютой.
for column in categorical_columns:
    mode_value = train_df[column].mode().iloc[0]

    train_df[column] = train_df[column].fillna(mode_value)
    test_df[column] = test_df[column].fillna(mode_value)

    print(f"Категориальный столбец {column}: мода = {mode_value}")

# Проверка заполнения пропусков.
print("\nПропуски ПОСЛЕ заполнения — обучающая выборка:")
print(train_df.isnull().sum())
print("\nПропуски ПОСЛЕ заполнения — тестовая выборка:")
print(test_df.isnull().sum())

print("\nРазмер обучающей выборки:", train_df.shape)
print("Размер тестовой выборки:", test_df.shape)

print("\nВсего пропусков в обучающей выборке:",
      train_df.isnull().sum().sum())
print("Всего пропусков в тестовой выборке:",
      test_df.isnull().sum().sum())

from sklearn.preprocessing import MinMaxScaler

# Масштабирование числовых признаков методом MinMaxScaler.
scaler = MinMaxScaler()

before_scaling = train_df[numeric_columns].head().copy()

# Параметры масштабирования определяем по обучающей выборке.
# Значения тестовой выборки могут выходить за диапазон [0, 1].
train_df[numeric_columns] = scaler.fit_transform(
    train_df[numeric_columns]
)

test_df[numeric_columns] = scaler.transform(
    test_df[numeric_columns]
)

print("\nПервые пять строк числовых признаков ДО нормализации:")
print(before_scaling.to_string())

print("\nТе же строки ПОСЛЕ нормализации:")
print(train_df[numeric_columns].head().to_string())

print("\nМинимумы и максимумы обучающей выборки:")
print(train_df[numeric_columns].agg(["min", "max"]))

# Из Cabin выделяем палубу и борт; полный номер каюты исключаем,
# чтобы избежать большого числа категорий при one-hot кодировании.
for table in [train_df, test_df]:

    cabin_parts = table["Cabin"].str.split("/")

    table["Deck"] = cabin_parts.str[0]

    table["Side"] = cabin_parts.str[2]

train_df = train_df.drop(columns=["Cabin"])
test_df = test_df.drop(columns=["Cabin"])

# Логические признаки и целевую переменную переводим в 0/1.
boolean_columns = ["CryoSleep", "VIP", "Transported"]

for column in boolean_columns:
    train_df[column] = train_df[column].astype(int)
    test_df[column] = test_df[column].astype(int)

# One-hot кодирование не задаёт искусственного порядка категорий.
columns_to_encode = ["HomePlanet", "Destination", "Deck", "Side"]

train_encoded = pd.get_dummies(
    train_df,
    columns=columns_to_encode,
    dtype=int
)

test_encoded = pd.get_dummies(
    test_df,
    columns=columns_to_encode,
    dtype=int
)

# Набор и порядок столбцов задаёт обучающая выборка.
# Неизвестным категориям в тесте соответствуют нулевые индикаторы.
test_encoded = test_encoded.reindex(
    columns=train_encoded.columns,
    fill_value=0
)

# Проверка структуры и отсутствия пропусков после кодирования.
print("\nПервые пять строк после кодирования категорий:")
print(train_encoded.head().to_string())

print("\nРазмеры таблиц после кодирования:")
print("Обучающая:", train_encoded.shape)
print("Тестовая:", test_encoded.shape)

print("\nОдинаковые столбцы в обеих таблицах:")
print(train_encoded.columns.equals(test_encoded.columns))

print("\nВсего пропусков после кодирования:")
print("Обучающая:", train_encoded.isnull().sum().sum())
print("Тестовая:", test_encoded.isnull().sum().sum())

# Сохранение результатов без индекса строк; исходный train.csv не меняется.
output_folder = Path(__file__).resolve().parent

train_encoded.to_csv(
    output_folder / "train_processed.csv",
    index=False
)

test_encoded.to_csv(
    output_folder / "test_processed.csv",
    index=False
)

print("\nОбработанные данные сохранены:")
print(output_folder / "train_processed.csv")
print(output_folder / "test_processed.csv")
