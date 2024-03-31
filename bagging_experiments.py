import numpy as np
import pandas as pd
import tensorflow as tf

import os

from sklearn.feature_extraction.text import CountVectorizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import Sequential, layers
from keras.regularizers import L2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_preparation import parse_data, prepare

data = list(parse_data('Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
original = pd.DataFrame.from_records(data).drop(["article_link"], axis=1)
prepared = prepare(original)

using_text = "good_words_2"
def numbers_only(inpu):
    df = inpu.loc[:, ~inpu.columns.str.startswith('good')]

    train, test = train_test_split(df, test_size=0.2, stratify=df["is_sarcastic"])

    X_train, y_train = train.drop("is_sarcastic", axis=1), train["is_sarcastic"]
    X_test, y_test = test.drop("is_sarcastic", axis=1), test["is_sarcastic"]

    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

    print(y_train)
    print(X_train.shape[1])

    model = Sequential([
        layers.Dense(X_train.shape[1]),
        layers.Dense(24, activation='relu', kernel_regularizer=L2(1e-3)),
        # layers.Dropout(0.1),
        layers.Dense(24, activation='relu', kernel_regularizer=L2(1e-3)),
        # layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=14
    )
    model.evaluate(X_test, y_test)

all_features = ["is_sarcastic", using_text, "punct", "percent_1", "percent_2", "percent_3"]
df = prepared[all_features]

list_of_texts = df[using_text].tolist()

# corpus_dict = {1: 'This is the first document.',
#           2: 'This is the second second document.',
#           3: 'And the third one.',
#           4: 'Is this the first document?'}
#
# df = pd.DataFrame(corpus_dict.items(), columns=['id', 'text'])

# list_of_texts = df['text'].tolist()
vectorizer = CountVectorizer(min_df=1, strip_accents="unicode", token_pattern=r"(?u)\b[A-z]{2,}\b", max_features=5000)
term_doc_matrix = vectorizer.fit_transform(list_of_texts)

dictionary = vectorizer.get_feature_names_out()
print(dictionary)
print(len(dictionary))


matrix = vectorizer.transform(df[using_text]).toarray()
full = np.concatenate((df[["is_sarcastic"]].to_numpy(), matrix), axis=1)



train, test = train_test_split(full, test_size=0.2, stratify=df["is_sarcastic"])

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]


print(X_train.shape)
model = Sequential([
        layers.Dense(X_train.shape[1]),
        layers.Dense(512, activation='relu', kernel_regularizer=L2(1e-3)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', kernel_regularizer=L2(1e-3)),
        layers.Dense(32, activation='relu', kernel_regularizer=L2(1e-3)),
        # layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])

model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        metrics=['accuracy']
)
history = model.fit(
        X_train, y_train,
        epochs=25
)
# model.evaluate(X_test, y_test)