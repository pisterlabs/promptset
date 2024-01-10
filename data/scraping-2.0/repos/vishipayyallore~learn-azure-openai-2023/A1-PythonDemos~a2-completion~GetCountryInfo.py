# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai

# Load config values
from dotenv import dotenv_values
config_details = dotenv_values(".env")

openai.api_type = "azure"
openai.api_base = config_details['OPENAI_API_BASE']
openai.api_version = config_details['OPENAI_API_VERSION']
openai.api_key = config_details["OPENAI_API_KEY"]

inputPrompt = "Please give me the country_name, capital_state, national_bird, country_population for India in JSON format"

response = openai.Completion.create(
    engine=config_details['COMPLETIONS_MODEL'],
    prompt=inputPrompt,
    temperature=1,
    max_tokens=300,
    top_p=0.5,
    frequency_penalty=0,
    presence_penalty=0,
    best_of=1,
    stop=None)

print(response.choices[0].text)