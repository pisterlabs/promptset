import os
import openai
openai.organization = "org-S5pmsKUFyqMX56WFMS9EE8KW"
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Model.list())
