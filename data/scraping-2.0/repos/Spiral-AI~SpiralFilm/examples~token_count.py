example_messages = [
    {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    {
        "role": "assistant",
        "content": "こんにちは。何かお手伝いしましょうか。",
    },
    {
        "role": "user",
        "content": "ぜひお願いします。とはいっても、特にお願いしたいことは無いんですけどね。例えばどんなことをお願いしても良いですか？",
    },
    {
        "role": "assistant",
        "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        "role": "user",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
    {
        "role": "assistant",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
]

from spiralfilm import FilmCore, FilmConfig
import openai


for model in [
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4",
]:
    print(model)
    # example token count from the function defined above

    # example token count from the OpenAI API
    response = openai.OpenAI().chat.completions.create(
        model=model,
        messages=example_messages,
        temperature=0,
        max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
    )

    tokens1 = response.usage.total_tokens - 1
    tokens2 = FilmCore("").num_tokens(example_messages, model=model)
    print(tokens1, tokens2)
