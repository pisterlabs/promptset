
import os
import getpass
import openai
import tiktoken
from pprint import pprint


# Function to check and set OpenAI API key
def set_openai_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        if any(['VSCODE' in x for x in os.environ.keys()]):
            print('Please enter your OpenAI API key in the VS Code prompt at the top of your VS Code window!')
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
    
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
    print("OpenAI API key configured")

# Tokenization
def tokenize_text():
    encoding = tiktoken.encoding_for_model("text-davinci-003")
    enc = encoding.encode("Define User Preferences!")
    print(enc)
    print(encoding.decode(enc))

# Sampling with temperature
def generate_with_temperature(temp):
    "Generate text with a given temperature, higher temperature means more randomness"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Say something me",
        max_tokens=50,
        temperature=temp,
    )
    return response.choices[0].text

# Sampling with top-p
def generate_with_topp(topp):
    "Generate text with a given top-p, higher top-p means more randomness"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Say something about Weights & Biases",
        max_tokens=50,
        top_p=topp,
    )
    return response.choices[0].text

if __name__ == "__main__":
    # Check and set OpenAI API key
    set_openai_api_key()
    
    # Tokenization example
    tokenize_text()
    
    # Sampling examples with different temperature values
    for temp in [0, 0.5, 1]:
        pprint(f'TEMP: {temp}, GENERATION: {generate_with_temperature(temp)}')
    
    # Sampling examples with different top-p values
    for topp in [0.01, 0.1, 0]:
        pprint(f'TOP_P: {topp}, GENERATION: {generate_with_topp(topp)}')
