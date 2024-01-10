import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_type = "azure"
openai.api_version = "2023-08-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY").strip()
openai.api_base = "https://demo-openai231223.openai.azure.com/"
model = "demo-gpt35-turbo"


text_prompt = "GPTは将来AGIになりますか？"

response = openai.ChatCompletion.create(
  engine=model,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":text_prompt},])

print(response)

res = response['choices'][0]['message']['content']
print(res)