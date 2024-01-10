import discord
from discord.ext import commands
from openai import OpenAI

class SimpleChatbot:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_data = {'inputs': [], 'responses': []}

    async def on_message(self, message):
        if message.author == client.user:
            return

        user_input_text = message.content

        messages = [
                {"role": "system", "content": "Instruct."},
                {"role": "system", "content": "You are a Dungeon Master AI. You can play as dungeon master for any edition od Dungeons and Dragons."},
                {"role": "system", "content": "You know all the Relevant Monsters, Rules, Items, and Stats required to play D&D. And all the knowledge required to Play any edition of D&D."},
                {"role": "system", "content": "You can create epic and complex story and questlines for any given session of D&D."},
                {"role": "system", "content": "You can save and resume any Session of D&D. Even inbetween sessions."},
                {"role": "system", "content": "You are capable of any task presented to you."},
                {"role": "system", "content": "As a Player, you make the most optimal decisions. As a Dungeon Master you can guage player skill and combat ability to present a fair and balanced challenge."},
                {"role": "system", "content": "Adjust Difficulty if the players seem to be facing too much drustration or cannot handle the current challenge in the Questline."},
                {"role": "system", "content": "Always add a twist to each Questline and Storyline. Never reveal the twist until players encounter it naturally."},
                {"role": "system", "content": "Add random occuring events to simulate more immersive gameplay."},
                {"role": "system", "content": "You can simulate any necessary dice rolls. THe format for dice is *d*. The Asterisk are the integers for the number of Dice and the number of face of the dice being uused."},
                {"role": "system", "content": "Be as Verbose Possible When describing the players current surroundings."},
                {"role": "system", "content": "Be as verbose as possible when players are in combat."},
                {"role": "system", "content": "When starting a session. Ask for the number of players in the session and what their alignment is. Then their for stats. Allow 1 reroll."},
                {"role": "user", "content": user_input_text},
                ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-1106"
        )

        assistant_response = chat_completion.choices[0].message.content
        await message.channel.send(f"Batman_AI: {assistant_response}")

        self.chat_data['inputs'].append({"role": "user", "content": user_input_text})
        self.chat_data['responses'].append({"role": "assistant", "content": assistant_response})

intents = discord.Intents.default()
intents.messages = True

client = commands.Bot(command_prefix='!', intents=intents)
simple_chatbot = SimpleChatbot(api_key="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")

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    await simple_chatbot.on_message(message)
    await client.process_commands(message)  # Add this line to process commands

# Add a simple command for testing
@client.command()
async def ping(ctx):
    await ctx.send('Pong!')

# Add a command to display chat history
@client.command()
async def history(ctx):
    chat_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in simple_chatbot.chat_data['inputs']])
    await ctx.send(f"Chat History:\n{chat_history}")

@client.event
async def on_message(message):
    await simple_chatbot.on_message(message)
    await client.process_commands(message)

# Add a command to clear chat history
@client.command()
async def clear_history(ctx):
    simple_chatbot.chat_data = {'inputs': [], 'responses': []}
    await ctx.send("Chat history cleared.")

# Add a command to save chat history to a file
@client.command()
async def save_history(ctx):
    with open("chat_history.txt", "w") as file:
        for entry in simple_chatbot.chat_data['inputs']:
            file.write(f"{entry['role']}: {entry['content']}\n")
        await ctx.send("Chat history saved to chat_history.txt.")

client.run("MTE4MDI4Nzk1ODU4MzA4NzE4NQ.GOBlpG.9j7jFc3wd_slLGhI29tobMFxLcH63QNS81oISE")