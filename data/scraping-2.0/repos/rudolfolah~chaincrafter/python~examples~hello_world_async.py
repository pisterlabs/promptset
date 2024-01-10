import asyncio

from chaincrafter import Chain, Prompt
from chaincrafter.models import OpenAiChat

chat_model = OpenAiChat(
    temperature=0.9,
    model_name="gpt-3.5-turbo",
    presence_penalty=0.1,
    frequency_penalty=0.2,
)


def make_chain(country):
    system_prompt = Prompt("You are a helpful assistant who responds to questions about the world")
    followup_prompt = Prompt("{city} sounds like a nice place to visit. What is the population of {city}?")
    hello_prompt = Prompt(f"Hello, what is the capital of {country}? Answer only with the city name.")
    return Chain(
        system_prompt,
        (hello_prompt, "city"),
        (followup_prompt, "followup_response"),
    )


async def main():
    chain_france = make_chain("France")
    chain_china = make_chain("China")
    results = await asyncio.gather(
        chain_france.async_run(chat_model),
        chain_china.async_run(chat_model),
    )
    for messages in results:
        for message in messages:
            print(f"{message['role']}: {message['content']}")

asyncio.run(main())
