#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://healthsummary.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ['OPENAI_API_KEY']

response = openai.ChatCompletion.create(
  engine="MSAILatest",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"hi the system seems tobe down?"},{"role":"user","content":"why this doesnt work at home?"}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response)