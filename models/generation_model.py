import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    GlobalMaxPooling1D,
)
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import numpy as np


# Определим функцию для создания модели
def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 1000, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(1000, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(total_words, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# Обучающие данные (нужно заполнить данными)
with open("data/text.txt", "r", encoding="utf-8") as f:
    TextData = f.read()


# Подготовка данных
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(TextData)
total_chars = len(tokenizer.word_index) + 1
max_sequence_len = 50


# Создание модели
def fit():

    input_sequences = []
    for i in range(0, len(TextData) - max_sequence_len, 1):
        sequence = TextData[i : i + max_sequence_len]
        input_sequences.append(sequence)

    input_sequences = tokenizer.texts_to_sequences(input_sequences)
    input_sequences = np.array(input_sequences)
    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_chars)

    # Создание и обучение модели
    model = create_model(total_chars, max_sequence_len)
    accuracy = 0
    epochs = 0
    while accuracy < 0.6:
        model.fit(xs, ys, epochs=1, verbose=1)
        loss, accuracy = model.evaluate(xs, ys, verbose=0)
        epochs += 1

    # Если ошибка не уменьшается на протяжении указанного количества эпох, то процесс обучения прерывается и модель инициализируется весами с самым низким показателем параметра "monitor"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # указывается параметр, по которому осуществляется ранняя остановка. Обычно это функция потреть на валидационном наборе (val_loss)
        patience=2,  # количество эпох по истечении которых закончится обучение, если показатели не улучшатся
        mode="min",  # указывает, в какую сторону должна быть улучшена ошибка
        restore_best_weights=True,  # если параметр установлен в true, то по окончании обучения модель будет инициализирована весами с самым низким показателем параметра "monitor"
    )

    # Сохраняет модель для дальнейшей загрузки
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="my_model",  # путь к папке, где будет сохранена модель
        monitor="val_loss",
        save_best_only=True,  # если параметр установлен в true, то сохраняется только лучшая модель
        mode="min",
    )

    # Сохраняет логи выполнения обучения, которые можно будет посмотреть в специальной среде TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir="log",  # путь к папке где будут сохранены логи
    )

    model.save("TextGenerator3000.h5")


fit()


# Генерация текста
def generate_text(seed_text, next_chars, model, max_sequence_len):
    generated_text = seed_text
    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )

        predicted_probs = model.predict(token_list)[0]
        predicted = np.argmax(predicted_probs)
        output_char = tokenizer.index_word.get(predicted, "")
        seed_text += output_char
        generated_text += output_char

    return generated_text
