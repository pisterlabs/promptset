import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import time
from openai import OpenAI


parser = argparse.ArgumentParser(description='Argument Parser for Generating Using GPT3.5-Turbo')
parser.add_argument('--test_loc', type=str, required=True, help='Location of the test file')
parser.add_argument('--out_file', type=str, required=True, help='Name of the file that will store the output generations')
args = parser.parse_args()

client = OpenAI()

r = pd.read_csv(f"{args.test_loc}")
words = list(r['Word'])
gold_riddles = list(r["Riddle"])

openai_generations = []

for i in tqdm(range(3,103)):
  word = words[i]
  msg = f"Can you create a riddle for the word: {word}?"
  completion = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      messages=[
        {"role": "user", "content": msg}
      ]
  )
  openai_generations.append(completion.choices[0].message.content)
  time.sleep(20)


openai_gen = {
    "Words":words,
    "GoldRiddle":gold_riddles,
    "GenRiddle":openai_generations
}


pd.DataFrame.from_dict(openai_gen).to_csv(f"{args.out_file}",index=False)