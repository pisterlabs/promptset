# =========================
#  Module: Summarize code
# =========================
import openai
import os
import argparse

def ask_openai(system: str, user: str) -> str:
    """
    Send a message to OpenAI and get the response.

    Parameters:
    - system (str): The system prompt to the assistant.
    - user (str): The user's prompt to the assistant.

    Returns:
    - str: The assistant's reply.
    """

    # Load your OpenAI API key from the environment variable
    openai.api_key = os.environ.get('OPENAI_API_KEY_ALTINN3_DEV')
    openai.api_base = os.environ['OPENAI_API_KEY_ALTINN3_DEV']

    # Ensure API key is present
    if not openai.api_key:
        raise ValueError("Missing value for environment variable 'OPENAI_API_KEY_ALTINN3_DEV'")

    # Define the message to be sent
    messages = [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': user[0:4096]}]

    # print(f'messages: {messages}')

    # Send the message to the OpenAI API
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=messages,
      temperature=0.1
    )

    # Extract and return the assistant's reply
    return response.choices[0].message['content']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask OpenAI based on a system prompt and user prompt from files.")
    parser.add_argument("system", help="Name of the file containing the system prompt.")
    parser.add_argument("user", help="Name of the file containing the user prompt.")
    parser.add_argument("output", help="Name of the file to save results.")
    args = parser.parse_args()

    with open(args.system, 'r') as sysfile:
        system_input = sysfile.read()
    
    with open(args.user, 'r') as userfile:
        user_input = userfile.read()
    
    response = ask_openai(system_input, user_input)

    with open(args.output, 'wt') as outputfile:
        outputfile.write(response)

    print(f"Assistant:\n\n {response}")