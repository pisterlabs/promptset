import os

import pandas as pd
import json
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
# Load the conversation data from the Excel file
file_path = r"C:\Users\kosmo\Desktop\Project\fine_tuning.xlsx"
df = pd.read_excel(file_path)

# Create a list of prompt and completion pairs
pairs = []
for i in range(len(df)):
    prompt = df.iloc[i]["prompt"]
    completion = df.iloc[i]["completion"]
    pair = {"prompt": prompt, "completion": completion}
    pairs.append(pair)

# Write the pairs to a JSONL file with UTF-8 encoding
output_file = r"C:\Users\kosmo\Desktop\Project\fine_tuning_data_20230730.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for pair in pairs:
        json.dump(pair, f, ensure_ascii=False)
        f.write("\n")
