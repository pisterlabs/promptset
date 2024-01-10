#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_ENDPOINT")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2022-12-01"

prompt = "Generate SaaS product name ideas for a copilot bot, that assists programmer to code better.\nSeed words: Develop, Intelligent, co-pilot, assit"

response = openai.Completion.create(
  engine="davinci-003",
  prompt=prompt,
  temperature=0.8,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None)

print(f"""\nPrompt: \n\n{prompt}\n\nResponse: \n{response.choices[0].text}""")