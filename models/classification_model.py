import numpy as np
import csv
import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import matplotlib.pyplot as plt

def plot_graphs(history, metric):

    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_" + metric])


def get_dataset_from_csv(file_path):
    """
    Извлекает набор данных из CSV файла.

    :param file_path: Путь к CSV файлу, содержащему данные
    :return: Кортеж из двух списков: комментарии и метки токсичности
    """
    dataset = ([], [])
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset[0].append(row["comment"])
            dataset[1].append(row["toxic"])
    return tf.data.Dataset.from_tensor_slices((dataset[0], dataset[1]))


train_dataset = get_dataset_from_csv("data/test_dataset.csv")

for example, label in train_dataset.take(1):
    print("text: ", example.numpy())
    print("label: ", label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (
    train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)

for example, label in train_dataset.take(1):
    print("texts: ", example.numpy()[:3])
    print()
    print("labels: ", label.numpy()[:3])


VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))


x = train_dataset.take(1).map(lambda text, label: text).numpy()[0]
y = train_dataset.take(1).map(lambda text, label: label).numpy()[0]

# Create a TextVectorization layer for input text
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=100
)

# Adapt the vectorization layer to the dataset
vectorize_layer.adapt(train_dataset.map(lambda text, label: text))

# Apply the vectorization layer to the input text
encoded_text = vectorize_layer(x)

print("Encoded Text Example: ", encoded_text.numpy()[:3])



