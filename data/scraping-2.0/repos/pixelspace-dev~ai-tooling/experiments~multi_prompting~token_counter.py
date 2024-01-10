import tiktoken
import streamlit as st


# calculates the number of tokens used and the percent remaining
def calculate_tokens_used(model) -> int:
    tokens_used = 0

    for message in st.session_state.chat:
        tokens_used += get_number_tokens_from_openai(message, "cl100k_base") - 1
    if st.session_state.set_new_prompt:
        tokens_used += get_number_tokens_from_openai(st.session_state.prompt, "cl100k_base") + 9
        # + 9 for "Sure, what's the user's inquiry?"

    max = how_many_tokens_remaining_as_int(tokens_used, model)

    percentage = round(100 - (tokens_used/max)*100, 2)

    return tokens_used, percentage


#get_number_tokens_from_openai takes an input message and an encoding, and returns the number of tokens used
#this uses tiktoken, from openai
def get_number_tokens_from_openai(message: str, encoding: str) -> int:
    tokens = tiktoken.get_encoding(encoding).encode(message)
    return len(tokens)


#takes number of tokens used and the model used, returns the number of tokens left to be used, which can be used in the response
def how_many_tokens_remaining_as_int(tokens_used: int, model: str) -> int:
    token_limits = {
        "gpt-4": 8192,
        "gpt-3.5-turbo-16k": 16384,
    }

    return token_limits.get(model, -1)





