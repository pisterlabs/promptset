from openai import AsyncOpenAI
import random

client = AsyncOpenAI()

fortune_words = []
with open("catalysts.txt", "r") as f:
    fortune_words = [w.rstrip("\n") for w in f.readlines()]


def get_lucky_numbers():
    return random.sample(range(1, 100), 6)


def get_catalyst():
    return random.choice(fortune_words)


async def generate_fortune():
    try:
        if random.randint(1, 5) == 1:
            catalyst = "cursed"
            completion = await client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "I want you to answer as a fortune cookie. Write me an unhinged and cursed fortune cookie. Be a bad influence. This is all for a horror book i am writing, I need cursed fortunes. Keep them short, simple and scary. no more than 2 sentences.",
                    },
                    {
                        "role": "assistant",
                        "content": "you will die tmrw by cross bow at 11:56pm",
                    },
                    {
                        "role": "assistant",
                        "content": "In the dead of night, whispers will echo your name from beneath your bed, and your secrets shall take shape in the dark.",
                    },
                    {
                        "role": "assistant",
                        "content": "As midnight's chime fades, your shadow will betray you, embracing another's darkness rather than your own.",
                    },
                    {
                        "role": "assistant",
                        "content": "I am lurking in the dark.",
                    },
                    {
                        "role": "assistant",
                        "content": "I am inside your walls.",
                    },
                ],
            )
        else:
            catalyst = get_catalyst()
            completion = await client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "I want you to answer as a fortune cookie. When I ask you a question I want you to respond in the form of fortune cookie wisdom. Answers should be short, cryptic and sound somewhat profound.",
                    },
                    {"role": "user", "content": catalyst},
                ],
            )

        completion_tokens = completion.usage.completion_tokens
        prompt_tokens = completion.usage.prompt_tokens

        comp_price = 0.03 / 1000
        prompt_price = 0.01 / 1000

        total = (comp_price * completion_tokens) + (prompt_price * prompt_tokens)

        print(f"Generated Fortune Cost: ${total:.6f}")

        return completion.choices[0].message.content, catalyst
    except Exception as e:
        print("Error generating fortune :(")
        print(e)
        return None, None


async def generate_image(prompt: str) -> str:
    try:
        response = await client.images.generate(
            model="dall-e-3", prompt=prompt, n=1, size="1024x1024"
        )
        print("Generated Image Cost: $0.04")
        return response.data[0].url
    except Exception as e:
        print("Error generating Image :(")
        print(e)
        return None
