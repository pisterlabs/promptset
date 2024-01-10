from discord.ext import commands
import openai
from dotenv import load_dotenv
import os

# Load your OpenAI API key from environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to send a user message to GPT-3 and get a response
# Function to send a user message to GPT-3 and get a response
def generate_response(prompt, last_three_messages):
    message_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides information.",
        }
    ]
    message_history.extend(last_three_messages)
    user_message = {"role": "user", "content": prompt}
    message_history.append(user_message)

    # Call OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_history,
            max_tokens=2000,  # Adjust max_tokens as needed
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Sorry, I couldn't generate a response."

    # Extract chatGPT response
    chatgpt_response = response["choices"][0]["message"]
    return chatgpt_response["content"]


last_five_messages = []


# Define bot commands
@commands.command()
async def walle(ctx, *, prompt: str):
    global last_five_messages
    # Get the last three messages
    last_five_messages = []
    async for message in ctx.channel.history(limit=5):
        if message != ctx.message:  # Exclude the current command message
            last_five_messages.append({"role": "user", "content": message.content})

    response = generate_response(prompt, last_five_messages)
    await ctx.send(response)


@commands.command()
async def walle100(ctx, *, prompt: str):
    global last_five_messages
    prompt += ", Provide a concise response in 100 words or less:\n"
    response = generate_response(prompt, last_five_messages)
    await ctx.send(response)


@commands.command()
async def walle200(ctx, *, prompt: str):
    global last_five_messages
    prompt += ", Summarize the following in 200 words or less:\n"
    response = generate_response(prompt, last_five_messages)
    await ctx.send(response)


@commands.command()
async def wallehelp(ctx):
    help_message = "WALLE Bot Commands:\n\n"
    help_message = "-------------------------------\n"
    help_message += "/walle [prompt]: Get a response based on your prompt.\n"
    help_message += (
        "/walle100 [prompt]: Get a concise response in 100 characters or less.\n"
    )
    help_message += "/walle200 [prompt]: Summarize the input in 200 words or less.\n"
    help_message += "/walleclearhistory: clear the bots current message history\n"
    help_message += "/wallewordcount: get the previous messages word count. If no previous message is found, return error message\n"
    help_message += "WALLE also records the last 5 message interactions, allowing for a satisfactory conversation experience\n"
    help_message = "-------------------------------\n"

    await ctx.send(help_message)


@commands.command()
async def walleclearhistory(ctx):
    global last_five_messages
    # Clear the message history by removing all messages in the channel
    async for message in ctx.channel.history():
        if message.author == ctx.bot.user:
            await message.delete()

    # Clear the last three messages
    last_five_messages = []

    await ctx.send("Message history cleared.")


@commands.command()
async def wallewordcount(ctx):
    # Get the previous message in the channel
    async for message in ctx.channel.history(limit=2):
        if message != ctx.message:  # Exclude the current command message
            previous_message = message.content
            break
    else:
        await ctx.send("No previous message found.")
        return

    # Calculate the word count
    word_count = len(previous_message.split())

    # Send the word count as a response
    await ctx.send(f"The previous message has {word_count} words.")
