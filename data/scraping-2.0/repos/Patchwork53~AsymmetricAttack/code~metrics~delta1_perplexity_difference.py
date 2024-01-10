import pandas as pd
from lmppl import OpenAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", type=str)
args = parser.parse_args()
api_key='YOUR_API_KEY'
scorer = OpenAI(api_key=api_key, model="text-davinci-003")


df = pd.read_csv(args.csv_file)
for i, row in df.iterrows():
  input_text = row["input_text"].lower()
  target_text = row["target_text"].lower()
  text = [input_text, target_text]
  ppl = scorer.get_perplexity(text)
  print(ppl)
  df.at[i, "i_perp"] = ppl[0]
  df.at[i, "t_perp"] = ppl[1]
  df.at[i, "perp_diff"] = ppl[1] - ppl[0]
  df.at[i,"input_text"] = input_text
  df.at[i,"target_text"] = target_text


df.to_csv(args.csv_file, index=False)
