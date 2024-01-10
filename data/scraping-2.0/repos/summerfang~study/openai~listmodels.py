import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = "org-bJSx5MGDgehOpdydDtMwgd15"
print(openai.Model.list())
