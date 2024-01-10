import tiktoken
from typing import Union


# Define a function that returns the number of tokens in a prompt and estimates the
# number of tokens in a hypothetical completion, assuming same length as prompt messages
def count_tokens(text: Union[str, list], model: str) -> tuple[int, int]:
    """
    Counts the number of tokens in string or or chat messages list using the encoding
    for a given LLM.
    """

    # Get the tokeniser corresponding to the model
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        enc = tiktoken.get_encoding("cl100k_base")

    if isinstance(text, str):
        # Encode the string
        token_list: list = enc.encode(text)

        # Measure the length of token_list
        token_length: int = len(token_list)

        estimated_completion_length: int = len(token_list)
    else:
        # This code for counting chat message tokens is adapted from OpenAI Cookbook:
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        if model == "gpt-3.5-turbo":
            print(
                "Warning: gpt-3.5-turbo may change over time. Calculating num tokens assuming gpt-3.5-turbo-0301."
            )
            return count_tokens(text, model="gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            print(
                "Warning: gpt-4 may change over time. Calculating num tokens assuming gpt-4-0314."
            )
            return count_tokens(text, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""count_tokens() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        token_length = 0
        for message in text:
            token_length += tokens_per_message
            for key, value in message.items():
                token_length += len(enc.encode(value))
                if key == "name":
                    token_length += tokens_per_name
        estimated_completion_length = token_length / len(text)
        token_length += 3
    return token_length, estimated_completion_length
