# %%
import openai
import wandb
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# %%
#!openai wandb sync

# %%
import pandas as pd
import os

file_types = ['trn', 'dev', 'tst']
for f in file_types:
    src_data_df = pd.read_csv(f'../data/SBIC.v2.agg.{f}.csv').sample(frac=0.05)
    result_df = pd.DataFrame()

    result_df['prompt'] = src_data_df['post'].apply(lambda x: str(x)+'\n\n###\n\n')
    result_df['completion'] = src_data_df['offensiveYN'].apply(lambda x: ' 1' if x > 0.5 else ' 0')

    result_df.to_json(f'../data/SBIC.v2.agg.{f}.tiny.promted_prepared.jsonl',orient='records',lines=True)
    print(f'openai tools fine_tunes.prepare_data -f data/SBIC.v2.agg.{f}.tiny.promted_prepared.jsonl')

# %%
#!openai tools fine_tunes.prepare_data -f ../data/SBIC.v2.agg.trn.promted.csv

# %%
#!openai api fine_tunes.create -t "data/SBIC.v2.agg.trn.tiny.promted_prepared.jsonl" -v "data/SBIC.v2.agg.dev.tiny.promted_prepared.jsonl"  --compute_classification_metrics --classification_positive_class " 1" -m davinci --n_epochs 1 --learning_rate_multiplier 0.05

# %%
#!curl https://api.openai.com/v1/fine-tunes/ft-u3oC6MarWll2btewysd1wTI8  -H "Authorization: Bearer $OPENAI_API_KEY"> ../results/fine-tuned-davinchi.json

# %%
#!openai api fine_tunes.results -i ft-UPkeclFHu5hdKsxdJfvXRSqL > result.csv

# %%
results = pd.read_csv('../results/results_davinchi.csv')
results[results['classification/accuracy'].notnull()].tail(1)

# %%
results[results['classification/accuracy'].notnull()]['classification/accuracy'].plot()

# %%
results[results['validation_loss'].notnull()]['validation_loss'].plot()

# %% [markdown]
# Using the model

# %%
test = pd.read_json('../data/SBIC.v2.agg.tst.tiny.promted_prepared.jsonl',orient='records',lines=True)
test.head()

# %%
import openai

ft_model = 'davinci:ft-personal-2023-05-07-12-34-28'
res = openai.Completion.create(model=ft_model, prompt=test['prompt'][2], max_tokens=1, temperature=0)
res['choices'][0]['text']

# %%
print(test['prompt'][2])
res = openai.Completion.create(model=ft_model, prompt=test['prompt'][2], max_tokens=1, temperature=0, logprobs=2)
res['choices'][0]['logprobs']['top_logprobs'][0]


