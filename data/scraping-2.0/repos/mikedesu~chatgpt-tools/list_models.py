import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

model_list = openai.Model.list()

#print(model_list)

for d in model_list.data:
    print(d.id)

