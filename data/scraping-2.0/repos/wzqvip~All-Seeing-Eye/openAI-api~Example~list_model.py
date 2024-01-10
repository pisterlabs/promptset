import os
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
models = openai.Model.list()

with open('models.json', 'w') as f:
    json.dump(models, f)
