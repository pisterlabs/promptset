import os
import openai
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")
question = sys.argv[1]

# Load knowledge base for AI as prompts & user questions

with open(".github/promptdoc.md", "r") as f:
    prompts = f.read()
    prompts += question

# OpenAI processing
response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompts,  # here is the prompt
    temperature=0.7,
    max_tokens=399,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

answer = response["choices"][0]["text"]

with open(".github/comment-template.md", 'a') as f:
    f.write(answer)
