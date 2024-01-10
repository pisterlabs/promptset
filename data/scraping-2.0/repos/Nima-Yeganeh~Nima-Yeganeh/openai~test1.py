import os
import openai
openai.organization = "Personal"
openai.api_key = os.getenv("")
openai.Model.list()
