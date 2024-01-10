import os
import openai

os.environ['OPENAI_API_KEY'] = "programando"
openai.api_key = os.environ['OPENAI_API_KEY']

question = "When did apple announced the Vision Pro?"
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          temperature=0,
                                          messages=[{"role": "user",
                                                     "content": question}])
print(completion["choices"][0]["message"]["content"])
