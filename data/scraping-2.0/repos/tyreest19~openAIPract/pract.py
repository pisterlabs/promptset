import os
import openai


api_key = "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"
openai.api_key = api_key
#returns a list of all OpenAI models
models = openai.Model.list()
print(models)
