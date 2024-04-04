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

model = tf.keras.models.load_model("../TrainedModels/test_model")
# tf.keras.utils.plot_model(model, "multi_input_model_good.png", show_shapes=True)
using_texts = ["good_words_2", "garbage_words"]


def my_predict(data):
    to_drop = [*using_texts, "headline"]
    if "is_sarcastic" in data:
        to_drop.append("is_sarcastic")

    predictions = model.predict({"my_values": data.drop(to_drop, axis=1),
                                 "garbage": data[using_texts[1]],
                                 "text": data[using_texts[0]]})
    results = pd.DataFrame(
        {"headline": data["headline"], "result": np.round(np.array(predictions).flatten(), 2)})
    #
    if "is_sarcastic" in data:
        results["is_sarcastic"] = data["is_sarcastic"]
    results = results.sort_values(by="result")
    return results


def get_results(headlines):
    prepared = prepare(headlines)

    all_features = ["punct", "percent_1", "percent_2", "percent_3", "headline"]
    all_features.extend(using_texts)
    if "is_sarcastic" in headlines:
        all_features.append("is_sarcastic")
    my_data = prepared[all_features]
    return my_predict(my_data)


onion = get_results(pd.read_table("../Data/new_onion_headlines.txt", names=["headline"]))
onion["is_sarcastic"] = 1
real = get_results(pd.read_table("../Data/new_real_articles.txt", names=["headline"]))
real["is_sarcastic"] = 0

original = pd.concat((onion, real))

print(prepare(original))

# data = list(parse_data('../Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
# original = pd.DataFrame.from_records(data).drop("article_link", axis=1)
# print(original)
nn_results = get_results(original)

nn_results["error"] = np.abs(nn_results["is_sarcastic"]-nn_results["result"])
print(nn_results.sort_values(by="error"))

for i in range(5, 95,10):
    table = nn_results.copy()
    table["result"] = np.where(table["result"] < i/100, 0, 1)

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
    print(f"{i/100} threshold: ", p, r, F)

# 0.35

print()
table = nn_results.copy()
table["result"] = np.where(table["result"] < 0.35, 0, 1)

print(table[table["result"]!=table["is_sarcastic"]])