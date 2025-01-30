import keras
import tensorflow as tf
from vector import to_vector, expr_to_vector

model = keras.models.load_model("./neural-network/models/model10.keras")

def predict(text, expr):
    x_text = tf.expand_dims(tf.constant(to_vector(text)), axis=0)
    x_expr = tf.expand_dims(tf.constant(expr_to_vector(expr)), axis=0)
    pred = model([x_text, x_expr], training=False)
    res = []
    pred = pred.numpy().flatten()
    for i, el in enumerate(pred):
        if el > 0.5:
            res.append(i)
    return res
