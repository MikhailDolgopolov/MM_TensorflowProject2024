import pandas as pd
import matplotlib.pyplot as plt
from data_preparation import parse_data, prepare

data = list(parse_data('Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
original = pd.DataFrame.from_records(data).drop("article_link", axis=1)
original.hist(column="is_sarcastic")
plt.show()