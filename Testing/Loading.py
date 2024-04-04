import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
import numpy as np
import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_preparation import parse_data, prepare

model = tf.keras.models.load_model("../TrainedModels/my_model")
using_texts = ["good_words_2", "garbage_words"]


def my_predict(data):
    to_drop = [*using_texts]
    to_drop.append("headline")
    predictions = model.predict({"my_values": data.drop(to_drop, axis=1),
                                 "garbage": data[using_texts[1]],
                                 "text": data[using_texts[0]]})
    results = pd.DataFrame(
        {"headline": data["headline"], "result": np.round(np.array(predictions).flatten(), 2)}).sort_values(
        by="result")

    return results


def get_results(path):
    articles = pd.read_table(path, names=["headline"])
    prepared = prepare(articles)

    all_features = ["punct", "percent_1", "percent_2", "percent_3", "headline"]
    all_features.extend(using_texts)
    my_data = prepared[all_features]
    return my_predict(my_data)


onion = get_results("../Data/new_onion_headlines.txt")
onion["is_sarcastic"] = 1
real = get_results("../Data/new_real_articles.txt")
real["is_sarcastic"] = 0

table = pd.concat((onion, real))

data = list(parse_data('../Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
original = pd.DataFrame.from_records(data).drop("article_link", axis=1)
prepared = prepare(original)

all_features = ["punct", "percent_1", "percent_2", "percent_3", "headline", "is_sarcastic"]
all_features.extend(using_texts)
nn_result = my_predict(prepared[all_features])
results = pd.DataFrame({"percent":[0], "precision":[0],"recall":[1], "F1":[0]})

for i in range(5, 96,10):
    # table = pd.concat((onion, real))
    table = nn_result.copy()
    table["result"] = np.where(table["result"] < i / 100, 0, 1)

    table["tp"] = np.where((table["result"] == 1) & (table["is_sarcastic"] == 1), 1, 0)
    table["tn"] = np.where((table["result"] == 0) & (table["is_sarcastic"] == 0), 1, 0)
    table["fn"] = np.where((table["is_sarcastic"] == 1) & (table["result"] == 0), 1, 0)
    table["fp"] = np.where((table["is_sarcastic"] == 0) & (table["result"] == 1), 1, 0)

    tp = np.count_nonzero(table["tp"])
    tn = np.count_nonzero(table["tn"])
    fp = np.count_nonzero(table["fp"])
    fn = np.count_nonzero(table["fn"])
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    F = 2 * (p * r) / (p + r)
    d = {"percent":i/100, "precision":p,"recall":r, "F1":F}
    results = results._append(d, ignore_index=True)
results=results.set_index("percent")

print(results)
