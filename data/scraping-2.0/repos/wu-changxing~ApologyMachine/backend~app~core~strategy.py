import openai
from dotenv import load_dotenv
import os
import toml

# Load environment variables
load_dotenv()

# Read OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Read prompt from TOML file
toml_data = toml.load("prompt.toml")



# Function to generate a similar apology using GPT-3
def gpt3_task(apology_to, apology_from, stragy):
    prompt_example = toml_data[stragy]["eg"]
    stragy_desc = toml_data[stragy]["desc"]

    # Formulate a new prompt for GPT-3
    new_prompt = f"Write an apology similar to this example: '{prompt_example}' and the apology should be at least 50 words long. apology to {apology_to} apology from {apology_from}, and you should use the following strategy: {stragy_desc}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": new_prompt}
        ]
    )
    return response.choices[0].message["content"]
