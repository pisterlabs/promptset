import os
import json
import openai
import tiktoken
import time
from vocabmaster import csv_handler


def format_prompt(language_to_learn, mother_tongue, words_to_translate):
    words_to_translate = "\n".join(words_to_translate)
    prompt = [
        {
            "role": "system",
            "content": """
            You are an expert at building vocabulary lists in a CSV file.
            You do NOT say anything else but the content of the CSV file.""",
        },
        {
            "role": "user",
            "content": f"""
            Translate the following {language_to_learn} words into {mother_tongue}
            and provide a CSV file with each row consisting of the {language_to_learn} word,
            its {mother_tongue} translations (if there are multiple translations possible,
            list them in the same column), and an example sentence in {language_to_learn}.

            Always give ONLY ONE example! The example HAS TO BE in {language_to_learn}!
            Separate each column with commas.
            For the translation column, ALWAYS give at least two or three possible translations!
            
            When you start a new row, you HAVE TO add a newline character.
            Below is the list of words to translate.
            The format should look like this:
            word,'translation1, translation2, translation3','example'.
            ---
            {words_to_translate}""",
        },
    ]
    return prompt


def chatgpt_request(
    prompt,
    model="gpt-3.5-turbo-0613",
    max_tokens=3600,
    n=1,
    temperature=0.7,
    stop=None,
    stream=False,
):
    start_time = time.monotonic_ns()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Make the API request
    response = openai.ChatCompletion.create(
        messages=prompt,
        model=model,
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
        stop=stop,
        stream=stream,
    )

    if stream:
        # Create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []

        # Iterate through the stream of events
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_messages.append(chunk_message)  # save the message
            print(chunk_message.get("content", ""), end="")  # stream the message
        print()
        response = collected_chunks

        # Save the time delay and text received
        response_time = (time.monotonic_ns() - start_time) / 1e9
        generated_text = "".join([m.get("content", "") for m in collected_messages])

    else:
        # Extract and save the generated response
        generated_text = response["choices"][0]["message"]["content"]

        # Save the time delay
        response_time = (time.monotonic_ns() - start_time) / 1e9

    return (
        generated_text,
        response_time,
        response,
    )


def num_tokens_from_string(string, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming"
            " gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming"
            " gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    elif model == "gpt-3.5-turbo-0613":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0613":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def estimated_cost(num_tokens, price_per_1k_tokens):
    """Returns the estimated cost of a number of tokens."""
    return f"{num_tokens / 1000 * price_per_1k_tokens:.6f}"


def estimate_prompt_cost(message):
    """Returns the estimated cost of a prompt."""
    num_tokens = num_tokens_from_messages(message)

    prices = {
        "gpt-3.5-turbo": 0.0015,
        "gpt-3.5-turbo-0613": 0.0015,
        "gpt-3.5-turbo-0613": 0.0015,
        "gpt-3.5-turbo-16k": 0.003,
        "gpt-4": 0.03,
        "gpt-4-0613": 0.03,
        "gpt-4-0613": 0.03,
        "gpt-4-32k": 0.06,
        "gpt-4-32k-0613": 0.06,
    }

    return {model: estimated_cost(num_tokens, price) for model, price in prices.items()}
