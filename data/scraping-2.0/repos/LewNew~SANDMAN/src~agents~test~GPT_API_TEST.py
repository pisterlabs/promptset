import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
import tiktoken


# Set your OpenAI API key


# Define a function to count tokens in a list of messages
def count_tokens(messages):
    # Combine all content into a single string
    text = "\n".join([message["content"] for message in messages])
    # Use the tiktoken library to count tokens
    tokenizer = tiktoken.Tokenizer()
    tokenizer.add_text(text)
    return tokenizer.count_tokens()


def chat(system, user_assistant, max_tokens):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, list), "`user_assistant` should be a list"

    system_msg = {"role": "system", "content": system}

    user_assistant_msgs = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
        for i, msg in enumerate(user_assistant)
    ]

    messages = [system_msg] + user_assistant_msgs

    # Check the token count
    token_count = count_tokens(messages)

    if token_count > max_tokens:
        raise ValueError(f"Total tokens ({token_count}) exceed the maximum allowed tokens ({max_tokens}).")

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=messages)

    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."

    return response["choices"][0]["message"]["content"]


# Define the maximum token limit you want to set
max_tokens_limit = 100  # Set your desired token limit here

# Example usage with token limit
try:
    response_fn_test = chat("You are a machine learning expert.", ["Explain what a neural network is."],
                            max_tokens_limit)
    print(response_fn_test)
except ValueError as e:
    print(e)
