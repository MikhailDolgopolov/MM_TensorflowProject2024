import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import json

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

data = list(parse_data('../Data/SarcasmInNews/Sarcasm_Headlines_Dataset_v2.json'))
df = pd.DataFrame.from_records(data)

print(df.head())
