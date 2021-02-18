""" data_analysis.py
    for generating pivot tables
    July 2020
"""

import argparse
import json

import pandas as pd

pd.set_option("display.max_rows", None)
parser = argparse.ArgumentParser(description="Analysis parser")
parser.add_argument("--filepath", default="output_default/test_stats.json", type=str)
args = parser.parse_args()

with open(args.filepath, 'r') as fp:
    data = json.load(fp)

num_entries = data.pop("num entries")
df = pd.DataFrame.from_dict(data, orient="index")
df["count"] = 1

values = ["train_acc", "test_acc"]
index = ["model", "dataset", "num_params"]

table = pd.pivot_table(df, index=index, aggfunc={"train_acc": ["mean", "std"],
                                                 "test_acc": ["mean", "std"],
                                                 "count": "count",
                                                 })
print(table)
