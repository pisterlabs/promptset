import os
import openai
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")

if sys.argv[1] == "AI-bot":
  model_name = "text-davinci-003"
elif sys.argv[1] == "Codex":
  model_name = "code-davinci-002"

# Load chat base for AI as prompts & user questions
with open("comments-no-quotes.txt", 'r') as f:
    prompts = f.read()

# OpenAI processing
if model_name == "code-davinci-002":
  response = openai.Completion.create(
    model=model_name,
    prompt=prompts,
    temperature=0.1,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=2,
    presence_penalty=1.1,
    stop=["###"]
  )
else:
  response = openai.Completion.create(
    model=model_name,
    prompt=prompts, # here is the prompt
    temperature=0.7,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
  )

answer = response["choices"][0]["text"]

with open(".github/comment-template.md", 'a') as f:
    f.write(answer)
