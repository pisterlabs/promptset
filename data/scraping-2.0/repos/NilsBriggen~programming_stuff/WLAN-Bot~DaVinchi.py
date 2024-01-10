import openai

openai.api_key = "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"

response = openai.Completion.create(
  model="code-davinci-002",
  prompt="\"\"\"\n1. Write Algorithm for Robot\n2. Robot should drive around in the whole house\n\"\"\"\n",
  temperature=0.9,
  max_tokens=5110,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0
)