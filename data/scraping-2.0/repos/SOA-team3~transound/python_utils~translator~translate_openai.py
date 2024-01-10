import os
import sys
from openai import OpenAI

# Get Ruby Input
ruby_input = sys.stdin.read()
text = ruby_input.split('\n')[0]
openai_api_key = ruby_input.split('\n')[1]
translate_language = ruby_input.split('\n')[2]

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Create an OpenAI client
client = OpenAI()

def translate_text(text, target_language):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": f"Translate the following text into {target_language}: {text}\n"
            },
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1
    )
    return response.choices[0].message.content

# Print the generated translation
translation = translate_text(text, translate_language)
print(translation)

# Old version # this is no longer supported in openai>=1.0.0
# import openai
# openai.api_key = openai_api_key
# prompt = f"Translate the following text into {translate_language}: {text}\n"
# response = openai.Completion.create(
#     engine = 'ada',
#     prompt = prompt,
#     max_tokens = 50,
#     n = 1,
#     temperature = 0.5
# )
# print(response.choices)