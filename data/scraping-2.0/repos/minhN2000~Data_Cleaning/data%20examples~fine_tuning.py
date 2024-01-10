import pandas as pd
import openai
import subprocess

df = pd.read_csv("data_example.csv")

prepared_data = df.loc[:,['prompt','completion']]
prepared_data.to_csv('prepared_data.csv',index=False)


subprocess.run('openai tools fine_tunes.prepare_data --file prepared_data.csv --quiet'.split())

subprocess.run('openai api fine_tunes.create --training_file prepared_data_prepared.jsonl --model code-davinci-002 --suffix "ExampleModel"'.split())
