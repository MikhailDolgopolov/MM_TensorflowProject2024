from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import tensorflow as tf

import os

from sklearn.naive_bayes import MultinomialNB

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import Sequential, layers
from keras.regularizers import L2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from data_preparation import parse_data, prepare

data = list(parse_data('Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
original = pd.DataFrame.from_records(data).drop(["article_link"], axis=1)
prepared = prepare(original)

def transform_model_data_w_count_vectorizer(preprocessed_text, Y_train,  X_test, Y_test):
    #vectorize dataset
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(preprocessed_text)

    #define model
    model = MultinomialNB(alpha=0.1)
    model.fit(vectorized_data, Y_train)

    #evaluate model
    predictions = model.predict(vectorizer.transform(X_test))
    accuracy = accuracy_score( Y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(Y_test, predictions)
    precision = precision_score(Y_test, predictions)

    print("Accuracy:",round(100*accuracy,2),'%')
    print("Balanced accuracy:",round(100*balanced_accuracy,2),'%')
    print("Precision:", round(100*precision,2),'%')
    return predictions


X_train, X_test, Y_train, Y_test = train_test_split(data['text'],
                                                    data['target'],
                                                    test_size=0.3,
                                                    random_state=0)

vectorizer = CountVectorizer()
vectorized_data = vectorizer.fit_transform(X_train)

#define model
model = MultinomialNB(alpha=0.1)
model.fit(vectorized_data, Y_train)

# predictions = model.predict(vectorizer.transform(X_test))
# accuracy = accuracy_score( Y_test, predictions)
# balanced_accuracy = balanced_accuracy_score(Y_test, predictions)
# precision = precision_score(Y_test, predictions)

transform_model_data_w_count_vectorizer(preprocessed_text_1, Y_train,  X_test, Y_test)