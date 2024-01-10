# Utility to print names of all available gpt models.

import openai

openai.api_key = open('key.txt', 'r').read().strip('\n')

models = openai.Model.list()

for model in models.data:
    model_id = model.id
    if 'gpt' in model_id:
        print(model_id)
