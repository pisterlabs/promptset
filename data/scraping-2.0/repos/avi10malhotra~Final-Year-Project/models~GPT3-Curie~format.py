"""
This script formats the training data according to the standards accepted by the GPT-3 API model.
"""

import openai
import pandas as pd

API_KEY = "sk-REDACTED"

openai.api_key = API_KEY

df = pd.read_csv("processed_dataset.csv", sep="\t")

df.columns = ["prompt", "completion"]

# add "->" to the prompt and "\n" to the completion
df["prompt"] = df["prompt"].apply(lambda x: x + " ->")
df["completion"] = df["completion"].apply(lambda x: x + "\n")
print(df.head())

# save the dataset
df.to_csv("training_data_prepared.jsonl", index=None)
