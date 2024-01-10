import os
import openai

openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")
a = openai.Model.list()

f = open("models.txt", "w")
f.write(str(a))
f.close()

