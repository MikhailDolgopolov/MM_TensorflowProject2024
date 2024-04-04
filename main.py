import pandas as pd
from keras import Sequential, layers
from keras.regularizers import L2
from sklearn.metrics import accuracy_score

from data_preparation import parse_data, prepare

data = list(parse_data('Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
original = pd.DataFrame.from_records(data).drop("article_link", axis=1)
using_text="good_words_2"
df = prepare(original)[["is_sarcastic", using_text]]


import keras
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


print(tf.config.list_physical_devices())

vocab_size = 6000

count_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='count'
)

train, test = train_test_split(df, test_size=0.2, stratify=df["is_sarcastic"])
train, validation = train_test_split(train, test_size=0.2, stratify=train["is_sarcastic"])
count_vectorizer.adapt(train[using_text].values)

training_dataset = tf.data.Dataset.from_tensor_slices(
    (train[using_text].values, train['is_sarcastic'].values)).batch(batch_size=32)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation[using_text].values, validation['is_sarcastic'].values)).batch(batch_size=32)
model = Sequential([
    count_vectorizer,
    layers.Dense(64, activation='relu', kernel_regularizer=L2(1e-3)),
    layers.Dropout(0.1),
    layers.Dense(64, activation='relu', kernel_regularizer=L2(1e-3)),
    # layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_focal_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

history = model.fit(
    training_dataset,
    steps_per_epoch=len(training_dataset),
    epochs=20,
    validation_data=validation_dataset,
    validation_steps=len(validation_dataset)
)

predictions = model.predict(test[using_text]).round()
print(accuracy_score(test['is_sarcastic'].values, predictions))