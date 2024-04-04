
import pandas as pd
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import Sequential, layers
from keras.regularizers import L2

from sklearn.model_selection import train_test_split
from data_preparation import parse_data, prepare

data = list(parse_data('Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
original = pd.DataFrame.from_records(data).drop(["article_link"], axis=1)
prepared = prepare(original)

using_texts = ["good_words_2", "garbage_words"]

all_features = ["is_sarcastic", "punct", "percent_1", "percent_2", "percent_3"]
x_features = all_features.extend(using_texts)

my_data = prepared[all_features]


imp_vocab_size = 10000

important_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=imp_vocab_size,
    output_mode='count'
)

garbage_vectorizer = tf.keras.layers.TextVectorization(
    output_mode='count'
)

train, test = train_test_split(my_data, test_size=0.2, stratify=my_data["is_sarcastic"])
important_vectorizer.adapt(train[using_texts[0]].values)
garbage_vectorizer.adapt(train[using_texts[1]].values)



X_train, y_train = train.drop("is_sarcastic", axis=1), train["is_sarcastic"]
X_test, y_test = test.drop("is_sarcastic", axis=1), test["is_sarcastic"]

print(X_test.shape)
print(y_test.shape)

values_input = tf.keras.Input(
    shape=(X_test.shape[1]-len(using_texts),), name="my_values"
)

text_input=tf.keras.layers.Input(dtype=tf.string, shape=(1,), name="text")
garbage_input=tf.keras.layers.Input(dtype=tf.string, shape=(1,), name="garbage")

imp_vocab_size = 5000

important_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=imp_vocab_size,
    output_mode='count', name="vectorizer"
)
important_vectorizer.adapt(X_train[using_texts[0]])
garbage_vectorizer.adapt(X_train[using_texts[1]])

important_vectorizer = important_vectorizer(text_input)
garbage_vectorizer = garbage_vectorizer(garbage_input)

important_vectorizer = layers.Dense(15, activation="elu",kernel_regularizer=L2(1e-2), name="1st_txt_layer")(important_vectorizer)
# important_vectorizer = layers.Dropout(0.2)(important_vectorizer)
# important_vectorizer = layers.Dense(20, activation="elu", name="2nd_txt_layer")(important_vectorizer)

garbage_vectorizer = layers.Dense(15, activation="elu", kernel_regularizer=L2(1e-2), name="1st_grb_layer")(garbage_vectorizer)

numbers_layer = layers.Dense(20, activation="elu", name="1st_number_layer")(values_input)

x = layers.concatenate([important_vectorizer, garbage_vectorizer, numbers_layer])
x = layers.Dense(20, activation="elu", name="Final_think")(x)

pred_layer = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=[text_input, garbage_input, values_input],
                       outputs =[pred_layer])

tf.keras.utils.plot_model(model, "multi_input_model.png", show_shapes=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)

print(X_train.columns)
print()
history = model.fit(
    {"my_values":X_train.drop(using_texts, axis=1),"garbage":X_train[using_texts[1]], "text":X_train[using_texts[0]]},
    y_train,
    epochs=12
)

model.save("TrainedModels/final_model", save_format="tf")
print("Trained: ")
loaded_vectorize_layer_model = tf.keras.models.load_model("TrainedModels/my_model")
model.evaluate({"my_values":X_test.drop(using_texts, axis=1),"garbage":X_test[using_texts[1]], "text":X_test[using_texts[0]]},
    y_test)
print("Loaded: ")
loaded_vectorize_layer_model.evaluate({"my_values":X_test.drop(using_texts, axis=1),"garbage":X_test[using_texts[1]], "text":X_test[using_texts[0]]},
    y_test)
# count_vectorizer = layers.Dense(20, activation="elu", name="1st_txt_layer")(count_vectorizer)
#
# numbers_layer = layers.Dense(20, activation="elu", name="1st_number_layer")(values_input)
#
# x = layers.concatenate([count_vectorizer, numbers_layer])
# x = layers.Dense(20, activation="elu", name="Final_think")(x)
#
# 0.86 0.81