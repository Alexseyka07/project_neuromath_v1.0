import keras
from keras import layers
from logic import filetool

import numpy as np
import tensorflow as tf

words = filetool.read_words()
exprs = filetool.read_exprs()
dataset = filetool.get_dataset(10)
d = dataset[0][0]

x_text = tf.constant(dataset[0][0])
x_expr = tf.constant(dataset[0][1])
y = tf.constant(dataset[1])

# input
text_input = keras.Input(shape=(None,), name="text_input")
math_expr_input = keras.Input(shape=(None,), name="math_expr_input")

# embedding
text_embedding = layers.Embedding(len(words), 512, name="text_embedding")(text_input)
expr_embedding = layers.Embedding(len(exprs),  40, name="expr_embedding")(math_expr_input)

# LSTM
text_features = layers.LSTM(1024, name="text_features")(text_embedding)
expr_features = layers.LSTM(80, name="expr_features")(expr_embedding)

# contact
features = layers.concatenate([text_features, expr_features])
# Dense
output_features = layers.Dense(44, activation="sigmoid", name="output")(features)

model = keras.Model(inputs=[text_input, math_expr_input], outputs=[output_features])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


model.fit(x=[x_text, x_expr], y=y, epochs=10, batch_size=20)


error = model.evaluate(x=[x_text, x_expr], y=y, batch_size=20)
result = model.predict(x=[x_text, x_expr], batch_size=20)

model.save("model10.keras")
model = keras.saving.load_model("model10.keras")
