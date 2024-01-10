import openai
from api_key import api_key
openai.api_key = api_key

print(openai.File.create(file=open("tag.jsonl", "rb"), purpose="fine-tune"))
