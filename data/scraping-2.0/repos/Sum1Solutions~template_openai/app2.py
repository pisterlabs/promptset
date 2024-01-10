# This is more like test app 2 to get this next stuff better understood.
import os
import openai
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Make sure it is set in the environment.")
    # Validate the format of the API key if needed
    # Add additional validation logic if necessary
    return api_key

def make_chat_completion_request(messages):
    openai.api_key = get_openai_api_key()

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return chat_completion

def run_chat_completion():
    messages = [{"role": "user", "content": "Hello world"}]  # Initialize with a user message
    termination_conditions = ["bye", "quit"]  # Define termination conditions
    max_response_length = 100  # Maximum number of tokens to display in a single print output

    num_tokens = 0  # Initialize the variable outside the loop

    while True:
        user_input = input("Enter your question here ('bye' or 'quit' to end session): ")
        messages.append({"role": "user", "content": user_input})

        if any(condition in user_input.lower() for condition in termination_conditions):
            break

        # Make API call
        chat_completion_result = make_chat_completion_request(messages)

        # Extract assistant response and process it if needed
        assistant_response = chat_completion_result["choices"][0]["message"]["content"]

        # Print assistant response
        print(f"Assistant:")

        # Split long responses into multiple print statements
        response_tokens = assistant_response.split()
        num_tokens = len(response_tokens)
        num_prints = (num_tokens - 1) // max_response_length + 1
        for i in range(num_prints):
            start = i * max_response_length
            end = (i + 1) * max_response_length
            print(' '.join(response_tokens[start:end]))

        messages.append({"role": "assistant", "content": assistant_response})

        # Ask whether to continue or quit if response is long
        if num_tokens > max_response_length:
            user_choice = input("There's more. Continue? (yes/no): ")
            if user_choice.lower() == "no":
                break

    # Calculate the total cost (this is just a rough estimate, consult OpenAI's pricing for accurate costs)
    total_tokens = sum(len(message["content"].split()) for message in messages if message["role"] == "assistant")
    cost_per_token = 0.0000002  # Replace with the actual cost per token from OpenAI
    total_cost = total_tokens * cost_per_token

    # Print metadata
    print(f"Total tokens: {total_tokens}")
    if num_tokens:
        print(f"Response length: {num_tokens} tokens")
    # Print total cost
    print(f"Total Cost: ${total_cost:.9f}")


if __name__ == "__main__":
    run_chat_completion()
