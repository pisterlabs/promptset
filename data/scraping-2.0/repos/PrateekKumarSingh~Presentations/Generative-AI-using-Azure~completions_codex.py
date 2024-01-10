#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_ENDPOINT")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2022-12-01"

prompt = "write a python function that take city name as a parameter, and return weather of the city using weather API"

response = openai.Completion.create(
  engine="codex",
  prompt=prompt,
  temperature=0.8,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None)

print(f"""\nPrompt: \n\n{prompt}\n\nResponse: \n{response.choices[0].text}""")