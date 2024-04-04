import re
import string

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

short_stopword_list = ("a, an, and, are, as, at, be, but, by, for, if, in, into, is, it, no, not, of, on, or, such, that, the, their, then, there,"
                       " these, they, this, to, was, will, with").split(
    ", ")

import requests

mid_list = requests.get(
    r"https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords")
mid_stopword_list = mid_list.text.split("\n")
# print(f"{len(mid_stopword_list)=}")

long_list = requests.get(r"https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt")
long_stopword_list = long_list.text.split("\n")


stopword_lists = [short_stopword_list, mid_stopword_list, long_stopword_list]

import json


def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l)


def prepare(df):
    def count(l1, l2): return sum([1 for x in l1 if x in l2])

    df["punct"] = df["headline"].apply(lambda s: count(s, string.punctuation))
    df["orig_word_count"] = df["headline"].apply(lambda t: len(t.split()))
    df["text"] = df["headline"].str.replace(r"[^\w\s]", '', regex=True)
    df["punct"] = df["punct"]/df["orig_word_count"]
    df["garbage_words"] = df["text"].apply(lambda x: ' '.join([word for word in x.split() if word in stopword_lists[-1]]))
    for i in range(len(stopword_lists)):
        col = f"good_words_{i + 1}"
        df[col] = df["text"].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stopword_lists[i])]))
        df[f"percent_{i + 1}"] = df[col].apply(lambda t: len(t.split())) / df["orig_word_count"]
    df = df.drop(["orig_word_count", "text"], axis=1)
    return df

def prepare_list(sentences, num):
    df = pd.DataFrame(sentences)
    return prepare(df)[f"good_words_{num}"]
