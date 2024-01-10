from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configuration Management: Load API key from environment variables
api_key = os.environ.get('OPENAI_API_KEY')
MODEL="gpt-4-1106-preview" # Update this to the desired model version
INSTRUCTION = "ScriptGPT specializes in creating and optimizing scripts for automation and integration tasks. It provides complete, ready-to-use scripts and offers suggestions for script improvements. Knowledgeable in various scripting languages, it addresses both simple and complex automation needs. While not executing scripts, Script Automator delivers secure, efficient, and best practice-oriented code solutions. If a user's request lacks specific details, it will ask for clarification to ensure accuracy and helpfulness. The interaction style is direct and practical, aiming to provide scripts efficiently. For users who need it, Script Automator can also include brief explanations or comments within the scripts, particularly useful for those less experienced in scripting."

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def scriptgpt_response(user_input, client):
    """Generate a response to the user input using OpenAI API."""
    user_input = user_input.lower()

    try:
        # Connect to the OpenAI API and get a response using the updated method
        stream = client.chat.completions.create(
            stream=True,
            model=MODEL,
            messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": user_input}
            ]
        )

    except Exception as e:
        return f"An error occurred: {e}"

    return stream

def chat():
    print("Welcome to ScriptGPT! What automation or integration script do you need? And in which language? Type 'bye' to exit.")

    while True:
        user_input = input("\033[92mYou: \033[0m")  # \033[92m is the ANSI escape code for green text, \033[0m resets the text color
        if user_input.lower() in ('bye', 'exit', 'quit'):
            print(f"\033[94mScriptGPT: \033[0mGoodbye! Have a nice day!")  # \033[94m is the ANSI escape code for blue text
            break
        stream = scriptgpt_response(user_input, client)
        print(f"\033[94mScriptGPT: \033[0m", end='') 
        for chunk in stream: 
            chunkContent = chunk.choices[0].delta.content 
            if chunkContent is None: 
                chunkContent = "\n" 
            print(chunkContent, end='')

if __name__ == "__main__":
    chat()