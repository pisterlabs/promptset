# Importing Libraries
import traceback
from telethon import TelegramClient, events

import config # Custom file containing configuration settings for the bot.
import openai # Python module that provides an interface to the OpenAI API.

from dotenv import load_dotenv

load_dotenv()


# openai.api_key = config.openai_key
openai.api_key = "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"
client = TelegramClient(None, config.API_ID, config.API_HASH).start(bot_token=config.BOT_TOKEN)


@client.on(events.NewMessage(pattern="(?i)/start"))
async def handle_start_command(event):
    try:

        async with client.conversation(await event.get_chat(), exclusive=True, timeout=600) as conv:
            history = []

            await conv.send_message("Hey, how's it going?")

            while True:
                # resp = (await conv.get_response()).raw_text
                # await conv.send_message(resp)

                user_input = (await conv.get_response()).raw_text
                user_input = f"You are Michael Jordan. {user_input}"
                history.append({"role":"user", "content": user_input})

                # Generate a chat completion using OpenAI API
                chat_completion = openai.ChatCompletion.create(
                    model=config.model_engine, # ID of the model to use.
                    messages=history, # The messages to generate chat completions for. This must be a list of dicts!
                    max_tokens=500, # The maximum number of tokens to generate in the completion.
                    n=1, # How many completions to generate for each prompt.
                    temperature=0.1 # Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
                )

                # Retrieve the response from the chat completion
                response = chat_completion.choices[0].message.content
                # Add the response to the chat history
                history.append({"role": "assistant", "content": response})
                await conv.send_message(response)

    except Exception:
        traceback.print_exc()
        

## Main function
if __name__ == "__main__":
    print("Bot Started...")    
    client.run_until_disconnected() # Start the bot!

