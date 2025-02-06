import datetime
import json
import tensorflow as tf
from tensorflow.keras.layers import (  # type: ignore
    Input,
    Embedding,
    Conv1D,
    Bidirectional,
    LSTM,
    MultiHeadAttention,
    GlobalMaxPooling1D,
    TextVectorization,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.utils import plot_model  # type: ignore
import re
import numpy as np


def mask_numbers(text):
    return re.sub(r"-?\d+\.?\d*", "[NUM]", text)


with open("data/new_math.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Применяем маскирование чисел ко всем текстам
for item in data:
    item["text"] = mask_numbers(item["text"])

# Разделяем данные на тексты и метки
texts = [item["text"] for item in data]
rules = [item["rules_id"] for item in data]

# Создаем слой для токенизации текста
max_tokens = 10000  # Максимальное количество слов в словаре
max_seq_length = 100  # Максимальная длина последовательности

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_sequence_length=max_seq_length,
    standardize="lower_and_strip_punctuation",  # Приводим к нижнему регистру и удаляем пунктуацию
)

# Адаптируем векторозатор к данным
vectorizer.adapt(texts)

# Преобразуем тексты в последовательности токенов
X = vectorizer(texts).numpy()

num_rules = 7
y = np.zeros((len(rules), num_rules), dtype=int)

for i, rule_ids in enumerate(rules):
    for rule_id in rule_ids:
        y[i, rule_id - 1] = 1

X_train, y_train = X, y
# Перемешиваем датасет
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

with open("data/teat_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Применяем маскирование чисел ко всем текстам
for item in test_data:
    item["text"] = mask_numbers(item["text"])

# Разделяем данные на тексты и метки
test_texts = [item["text"] for item in test_data]
test_rules = [item["rules_id"] for item in test_data]

# Преобразуем тексты в последовательности токенов
X_test = vectorizer(test_texts).numpy()

y_test = np.zeros((len(test_rules), num_rules), dtype=int)

for i, rule_ids in enumerate(test_rules):
    for rule_id in rule_ids:
        y_test[i, rule_id - 1] = 1

X_test, y_test = X_test, y_test

# Перемешиваем датасет
indices = np.arange(len(X_test))
np.random.shuffle(indices)
X_test = X_test[indices]
y_test = y_test[indices]


def create_model(max_seq_length=100, vocab_size=10000, num_rules=7, model_name="model"):
    inputs = Input(shape=(max_seq_length,))

    x = Embedding(vocab_size, 128)(inputs)

    x = Conv1D(128, 3, activation="relu", padding="same")(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_rules, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "AUC"],
    )
    model.summary()

    plot_model(model, to_file=f"{model_name}_shema.png", show_shapes=True)

    return model


model = create_model()


# Save the model architecture to a file


# Evaluate the model


# Log the training data to TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=2, mode="min", restore_best_weights=True
)
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=12,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback],
)
loss, accuracy, precision, auc = model.evaluate(X_train, y_train)
print(f"Training accuracy: {accuracy:.2f}")
# Пример задачи
while True:
    new_task = input("Введите новую задачу: ")
    new_task_vectorized = vectorizer([mask_numbers(new_task)]).numpy()

    # Предсказание
    predictions = model.predict(new_task_vectorized)
    predicted_rules = (
        np.where(predictions > 0.5)[1] + 1
    )  # Преобразуем индексы в номера правил
    print("Предсказанные правила:", predicted_rules)
