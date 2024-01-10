import os
import openai
openai.organization = "org-g6npB75HTdV0XQlZ7UtP0HFv"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()
print("hello world!")
