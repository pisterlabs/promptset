import tiktoken


"""Adapted from OpenAI cookbook:
https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
"""


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
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


# transform previous table into a dictionary
model_prices_in_cents_per_token = {
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-1106-vision-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
}


def get_input_costs_in_dollar(model_name, messages):
    """Return the cost per token for a given model."""
    if model_name in model_prices_in_cents_per_token:
        price_per_token = model_prices_in_cents_per_token[model_name]["input"]
    else:
        raise NotImplementedError(
            f"""get_input_costs() is not implemented for model {model_name}."""
        )

    num_input_tokens = num_tokens_from_messages(messages, model_name)

    return num_input_tokens * price_per_token / 100


def get_output_costs_in_dollar(model_name, messages):
    """Return the cost per token for a given model."""
    if model_name in model_prices_in_cents_per_token:
        price_per_token = model_prices_in_cents_per_token[model_name]["output"]
    else:
        raise NotImplementedError(
            f"""get_output_costs() is not implemented for model {model_name}."""
        )

    num_output_tokens = num_tokens_from_messages(messages, model_name)

    return num_output_tokens * price_per_token / 100


def get_job_input_cost_in_dollar(jobs):
    total_price = 0
    for job in jobs:
        total_price += get_input_costs_in_dollar(job["model"], job["messages"])

    return total_price


def get_job_output_cost_in_dollar(jobs):
    total_price = 0
    for job in jobs:
        total_price += get_output_costs_in_dollar(job["model"], job["messages"])

    return total_price
