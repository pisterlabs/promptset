import openai
import os

# set the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# define the prompt
prompt = "Write a short story about a detective investigating a crime."

# set the parameters for text generation
model_engine = "text-davinci-002"
temperature = 0.7
max_tokens = 50

# generate text from the prompt
response = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature
)

# print the generated text
print(response.choices[0].text)
