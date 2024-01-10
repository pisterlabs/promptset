import discord
from discord.ext import commands
import openai
import ipdb

"""
ft:gpt-3.5-turbo-0613:personal::8AQSKwfK is most up to date lucy
ft:gpt-3.5-turbo-0613:personal::8ASv0N4p me
"""


def call_lucy_gpt(messages):
    try:
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::8AQSKwfK",
            messages=messages,
            max_tokens=100,
            temperature=0.8,
        )
        # ipdb.set_trace()

        return completion.choices[0]["message"]["content"]
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_KEY")

    # Create a new instance of a bot.
    intents = discord.Intents.default()
    intents.messages = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_message(message):
        # Ignore messages from the bot itself
        if (
            message.author == bot.user
            or message.channel.id != 1159142635672449094
            or message.author.id == 1163665414052642897
        ):
            print("wrong channel")
            return
        context = {"messages": []}
        messages = [{"role": "system", "content": "You are Lucy Bot."}]

        async for entry in message.channel.history(
            limit=25,
        ):
            message_text = entry.content
            author_id = entry.author.id
            channel_id = entry.channel.id
            if author_id == 1159266154372677682:
                messages.append({"role": "assistant", "content": message_text})
            else:
                messages.append({"role": "user", "content": message_text})
        messages.reverse()
        # ipdb.set_trace()
        print("It's the right channel!")
        message_text = call_lucy_gpt(messages)
        print(message_text)
        if message_text:
            await message.channel.send(message_text.replace("@", "#"))

    # Run the bot with its token.
    bot.run(os.getenv("DISCORD_TOKEN"))
