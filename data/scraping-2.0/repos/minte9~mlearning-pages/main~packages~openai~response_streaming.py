import openai
import os, sys

# Setup OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# User question
print("Please enter your question (press Enter to submit):")
question = input()

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": question}],
    max_tokens=256,
    n=1,
    stop=None,
    temperature=0.7,
    stream=True
)

for chunk in response:
    content = chunk["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
    
print("\n")