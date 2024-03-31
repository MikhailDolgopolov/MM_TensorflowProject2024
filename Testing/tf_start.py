import keras
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)
print(tf.config.list_physical_devices())

X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42, cluster_std=1)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h=1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

all_x = xx.flatten()
all_y = yy.flatten()

m = np.array([all_x, all_y]).T
print(m.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train, s=25, edgecolor="k")
# plt.scatter(X_train, y_train)
# plt.show()
# print(X_test, y_test)
model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(22, activation='relu'),
        # tf.keras.layers.Dropout(0.4),
        # tf.keras.layers.Dense(22, activation='relu'),
        # tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError("sum_over_batch_size")
# loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=loss_fn,
              metrics=["accuracy"])
              # loss=loss_fn,
              # metrics=['accuracy'])


cm = plt.cm.RdBu
model.fit(X_train, y_train, epochs=50)
print("My test")
model.evaluate(X_test,  y_test, verbose=2)
preds = model.predict(X_test)
all = model.predict(m)
plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=preds, s=(y_test+2)*20)
plt.contourf(xx, yy, all.reshape(xx.shape), cmap=cm, alpha=0.3, levels=20)
# plt.scatter(X_test, preds, c="red", marker="o", s=25)
# plt.scatter(X_test, y_test, c="blue", marker="o", s=25)
plt.show()