import openai
import tiktoken
from openai import AsyncOpenAI

openai_async = AsyncOpenAI()
openai_completions = openai_async.chat.completions

models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-1106-preview",
]


def num_tokens_from_messages(messages, model="gpt-4-1106-preview"):
    try:
        encoding = tiktoken.encoding_for_model(model)

    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in models:
        tokens_per_message = 3
        tokens_per_name = 1

    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1

    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")

    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")

    else:
        raise NotImplementedError(f"""Not implemented for {model}.""")

    num_tokens = 0

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


examples = [
    {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Things working well together will increase revenue.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
]


async def tokenize():
    for model in models:
        print(f"{num_tokens_from_messages(examples, model)} prompt tokens counted.")

        response = await openai_completions.create(
            model=model, messages=examples, temperature=0, max_tokens=1
        )
        print(f'{response["usage"]["prompt_tokens"]} prompt tokens counted.')
