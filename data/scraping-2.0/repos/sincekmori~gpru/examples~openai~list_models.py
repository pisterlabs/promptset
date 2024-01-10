import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

model_list = api.list_models()
print(model_list.ids)
# Example output:
# ["babbage", "davinci", ..., "text-curie:001", "text-babbage:001"]
