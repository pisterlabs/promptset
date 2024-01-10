from chaincrafter import Chain, Prompt
from chaincrafter.models import OpenAiChat

chat_model = OpenAiChat(
    temperature=0.9,
    model_name="gpt-3.5-turbo",
    presence_penalty=0.1,
    frequency_penalty=0.2,
)
system_prompt = Prompt("You are a helpful assistant who responds to questions about the world")
hello_prompt = Prompt("Hello, what is the capital of France? Answer only with the city name.")
followup_prompt = Prompt("{city} sounds like a nice place to visit. What is the population of {city}?")
chain = Chain(
    system_prompt,
    (hello_prompt, "city"),
    (followup_prompt, "followup_response"),
)
messages = chain.run(chat_model)
for message in messages:
    print(f"{message['role']}: {message['content']}")
