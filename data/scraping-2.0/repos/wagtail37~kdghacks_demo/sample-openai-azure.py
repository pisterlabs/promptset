#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "api_base"#https://OPENAI_MODEL_NAME.openai.azure.com/
openai.api_version = "2023-03-15-preview"
openai.api_key = "api_key"

#質問の設定
content = "プロンプト"

response = openai.ChatCompletion.create(
  engine="engine",#DEPLOYED_MODEL_NAME
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":content},],
  temperature=0,
  max_tokens=800,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response['choices'][0]['message']['content'])