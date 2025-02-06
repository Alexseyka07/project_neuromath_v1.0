import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import numpy as np

# Генерация данных
X = np.array(np.arange(-100, 100), dtype=float)
y = np.array([x**2 for x in X], dtype=float)

# Разделение данных на обучающую и тестовую выборки
X_train = X[:160]
y_train = y[:160]
X_test = X[160:]
y_test = y[160:]

model = tf.keras.Sequential(
    [
        layers.Dense(units=64, activation="relu", input_shape=[1]),
        layers.Dense(units=64, activation="relu"),
        layers.Dense(units=1),
    ]
)

# Компиляция модели
model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

tf.keras.models.save_model(model, "math_model.h5")

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

predictions = model.predict(X_test)
for i in range(5):
    print(f"Input: {X_test[i]}, Predicted: {predictions[i][0]}, Actual: {y_test[i]}")

loaded_model = tf.keras.models.load_model("math_model.h5")

while True:
    x = float(input("Enter a number: "))    
    prediction = loaded_model.predict(np.array([x]))
    print(f"Input: {x}, Predicted: {prediction[0][0]}")
    