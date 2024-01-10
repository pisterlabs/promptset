import os
import openai
from langchain.llms import OpenAI

# Set up OpenAI API credentials
openai.organization = "org-9PKnn7txwxhPDTNvTe3ZL164"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the Langchain OpenAI LLM with a temperature of 0.9
llm = OpenAI(temperature=0.9)

# Prompt user for input
text = input("Enter a prompt for a movie plot summary: ")

# Generate movie plot summary using Langchain OpenAI LLM
generated_text = llm(text)

# Send generated text to OpenAI API to check for coherence
response = openai.Completion.create(
    engine="davinci",
    prompt=generated_text,
    max_tokens=100
)

# Print final movie plot summary
print("\nMovie Plot Summary:\n" + "-" * 20)
print(response.choices[0].text.strip())
