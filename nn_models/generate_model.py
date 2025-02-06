import tensorflow as tf 
import numpy as np

import os  
import json
import time  

def get_text_json(path_to_file):
    with open(path_to_file, "r") as file:
        text = json.load(file)
    return text

def get_text_to_path(path_to_file, name):

    path_to_file = tf.keras.utils.get_file(
        name,
        path_to_file
    )
    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    return text


json_text = get_text_json("data/dataset_mat.json")
input_text = [x["общий вид задачи"] for x in json_text]
target_text = [" ".join(x["правила"]) for x in json_text]

vocab = sorted(set("".join(map(str, input_text))) | set("".join(map(str, target_text))))

# Создаем слой для преобразования символов в их числовые идентификаторы
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab),
    mask_token=None,
)

# Создаем слой для обратного преобразования числовых идентификаторов в символы
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), 
    invert=True, 
    mask_token=None
)

# Функция для преобразования идентификаторов в текст
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Преобразуем текст в последовательность идентификаторов для input_text и target_text
input_ids = ids_from_chars(tf.strings.unicode_split(input_text, "UTF-8"))
target_ids = ids_from_chars(tf.strings.unicode_split(target_text, "UTF-8"))

# Создаем датасет из последовательности идентификаторов
input_ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids)
target_ids_dataset = tf.data.Dataset.from_tensor_slices(target_ids)

# Объединяем input_ids_dataset и target_ids_dataset в один датасет
dataset = tf.data.Dataset.zip((input_ids_dataset, target_ids_dataset))

input_example_list, target_example_list = [], []
for input_example, target_example in dataset:
    input_example_list.append(input_example)
    target_example_list.append(target_example)

# Размер батча
BATCH_SIZE = 12

# Размер буфера для перемешивания датасета
BUFFER_SIZE = 200

# Перемешиваем, разбиваем на батчи и подготавливаем датасет
dataset = (
    dataset.shuffle(BUFFER_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

for input_example_batch, target_example_batch in dataset.take(1):
    print("Input shape: ", input_example_batch.shape)
    print("Target shape: ", target_example_batch.shape)
    print("Input: ", text_from_ids(input_example_batch).numpy().decode("utf-8"))
    print("Target: ", text_from_ids(target_example_batch).numpy().decode("utf-8"))


# Длина словаря в слое StringLookup
vocab_size = len(ids_from_chars.get_vocabulary())

# Размерность встраивания
embedding_dim = 256

# Количество блоков RNN
rnn_units = 1024

# Определяем модель
class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.GRU = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            batch_size = tf.shape(x)[1]
            states = self.GRU.get_initial_state(batch_size=batch_size)

        x, states, = self.GRU(x, initial_state=states, training=training)
        x = self.dense(x, training=training)


        if return_state:
            return x, states
        else:
            return x

# Создаем экземпляр модели
model = MyModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)

# Пример предсказания модели
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(
        example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)"
    )

# Выводим сводку модели
model.summary()

# Генерируем случайные индексы
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1)

# Выводим входные данные и предсказанные символы
print("Input:\n", text_from_ids(input_example_batch).numpy().decode("utf-8"))
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy().decode("utf-8"))

# Определяем функцию потерь
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# target_example_batch = tf.squeeze(target_example_batch, axis=-1)

# # Вычисляем среднее значение потерь для примера
# # example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
# # print(
#     "Prediction shape: ",
#     example_batch_predictions.shape,
#     " # (batch_size, sequence_length, vocab_size)",
# )
# print("Mean loss:        ", example_batch_mean_loss)

# # Вычисляем экспоненту от среднего значения потерь
# tf.exp(example_batch_mean_loss).numpy()

# Компилируем модель
model.compile(optimizer="adam", loss=loss)

# Директория для сохранения контрольных точек
checkpoint_dir = "./training_checkpoints"
# Имя файлов контрольных точек
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Определяем callback для сохранения контрольных точек
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, f"ckpt_{{epoch}}.weights.h5"),
    save_weights_only=True,
)

# Количество эпох обучения
EPOCHS = 5
# Обучаем модель
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback], verbose=-1)

# Определяем класс для генерации текста по одному шагу
