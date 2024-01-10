import os
import discord
import openai

# Create a new Discord client
client = discord.Client(intents=discord.Intents.default())

# Set up the OpenAI client
openai.api_key = "YOUR OPENAI API KEY"

# Define a message event handler
@client.event
async def on_message(message):
    server_name = message.guild.name
    bot_display_name = client.user.display_name

    # If the message is from the bot itself, ignore it
    if message.author == client.user:
        return

    # If the message mentions the bot
    if message.mention_everyone or client.user in message.mentions:
      # Remove bot mention from prompt
        prompt_start = message.content.split()[0]
        prompt_text = message.content.replace(prompt_start,"")
        prompt_text = prompt_text.lstrip()
      # Edit Instruction text
        instruction_text = f"You are a bot named {bot_display_name} and you are in a Discord server named {server_name}. You have to respond to every prompt as accurately as possible given the following prompt: \n"
        prompt_text = instruction_text + prompt_text
        # Create a new Completion object for OpenAI
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt_text,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # Get the response text from the Completion object
        response_text = completion.choices[0].text
        try:
            await message.channel.send(response_text)
        except:
          # Seldom you can get errors in the response text, this skips it in case that happens
            await message.channel.send("An error ocurred...")

# Start the bot
try:
    client.run("YOUR DISCORD TOKEN")
except:
    os.system("kill 1")
