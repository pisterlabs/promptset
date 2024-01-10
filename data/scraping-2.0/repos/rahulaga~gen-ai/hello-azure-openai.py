import os
import openai

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

#log details
openai.log='debug'

response = openai.ChatCompletion.create(
  engine="gpt35",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"What other plants can grow well where grapes thrive? Think step by step"}],
  temperature=0.7,
  max_tokens=800)

print(response)
