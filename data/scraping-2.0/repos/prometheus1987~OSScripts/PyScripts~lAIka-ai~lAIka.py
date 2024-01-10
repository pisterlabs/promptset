import openai
import os
from dotenv import load_dotenv


# Replace '
# 
# YOUR_API_KEY' with your OpenAI API key

def load_dotenv():

    api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI API client
    openai.api_key = api_key

banner = """

██████╗░██████╗░░█████╗░░░░░░██╗███████╗██╗░░██╗████████╗
██╔══██╗██╔══██╗██╔══██╗░░░░░██║██╔════╝██║░██╔╝╚══██╔══╝
██████╔╝██████╔╝██║░░██║░░░░░██║█████╗░░█████═╝░░░░██║░░░
██╔═══╝░██╔══██╗██║░░██║██╗░░██║██╔══╝░░██╔═██╗░░░░██║░░░
██║░░░░░██║░░██║╚█████╔╝╚█████╔╝███████╗██║░╚██╗░░░██║░░░
╚═╝░░░░░╚═╝░░╚═╝░╚════╝░░╚════╝░╚══════╝╚═╝░░╚═╝░░░╚═╝░░░

██████╗░░█████╗░░██████╗██████╗░██╗░░░██╗████████╗███╗░░██╗██╗██╗░░██╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗██║░░░██║╚══██╔══╝████╗░██║██║██║░██╔╝
██████╔╝███████║╚█████╗░██████╔╝██║░░░██║░░░██║░░░██╔██╗██║██║█████═╝░
██╔══██╗██╔══██║░╚═══██╗██╔═══╝░██║░░░██║░░░██║░░░██║╚████║██║██╔═██╗░
██║░░██║██║░░██║██████╔╝██║░░░░░╚██████╔╝░░░██║░░░██║░╚███║██║██║░╚██╗
╚═╝░░╚═╝╚═╝░░╚═╝╚═════╝░╚═╝░░░░░░╚═════╝░░░░╚═╝░░░╚═╝░░╚══╝╚═╝╚═╝░░╚═╝
"""

def chat_with_gpt(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",  # Choose an appropriate engine
            prompt=prompt,
            max_tokens=50,  # Adjust the maximum number of tokens in the response
            stop=None  # Specify a list of strings to stop the completion
        )

        return response.choices[0].text.strip()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    print(banner)
    load_dotenv()
    print(api)
    user_input = input("You: ")
    while user_input.lower() != "exit":
        response = chat_with_gpt(user_input)
        print(f"ChatGPT: {response}")
        user_input = input("You: ")