import os
import openai

import env

openai.organization = "org-HExHkFBKWnhp47YeDnAi3yQS"
openai.api_key = os.getenv("OPENAI_API_KEY")
models = openai.Model.list()
print("models: ", models)