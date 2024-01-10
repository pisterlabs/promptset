from decouple import config
from openai import OpenAI

client = OpenAI(api_key=config("OPENAI_API_KEY"))


def consistency_printer(response):
    response_content = response.choices[0].message.content
    system_fingerprint = response.system_fingerprint
    print(f"\033[94m {response_content} \033[0m")
    print(f"\033[92m {system_fingerprint} \033[0m")


def bedtime_stories(query, seed=None, model="gpt-3.5-turbo-1106"):
    messages = [
        {
            "role": "system",
            "content": "You make up fun children's stories according to the user request. The stories are only 100 characters long.",
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        temperature=0.7,
        stop=["\n"],
    )
    consistency_printer(response)


# for i in range(3):
#     bedtime_stories(
#         "Tell me a story about a unicorn in space.",
#         seed=2424,
#         model="gpt-4-1106-preview",
#     )


def fruit_gpt(query, seed=None, temperature=0.2):
    messages = [
        {
            "role": "system",
            "content": "You are the fruitclopedia. Users name a fruit and you give information.",
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        seed=seed,
        temperature=temperature,
        stop=["\n"],
    )
    consistency_printer(response)


for i in range(3):
    fruit_gpt(
        "Grapefruit.",
        seed=123,
        temperature=0,
    )
