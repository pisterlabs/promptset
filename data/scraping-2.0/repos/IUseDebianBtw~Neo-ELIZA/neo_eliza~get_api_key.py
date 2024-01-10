import openai
import sys

def get_api_key():
    openai.api_key = input("Please enter your OpenAI API key (or type 'q'/'quit' to exit): ")

    if openai.api_key.lower() in ['q', 'quit']:
        sys.exit(0)
        
    try:
        openai.Model.list()
    except openai.error.OpenAIError as e:
        if 'authentication' in str(e).lower():
            print("Error: Invalid API key. Please enter a valid key.")
        else:
            print(f"An error occurred: {e}")
