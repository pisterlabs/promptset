import asyncio
from telethon import TelegramClient, errors
import openai
from config import (YOUR_API_ID, YOUR_API_HASH, OPENAI_API_KEY, YOUR_GROUP_LINK, GROUP_ID)

# Setup
openai.api_key = OPENAI_API_KEY

# Telethon client
client = TelegramClient('anon', YOUR_API_ID, YOUR_API_HASH)

async def process_last_messages():
    try:
        # Resolve the entity before fetching messages
        group_entity = await client.get_entity(YOUR_GROUP_LINK)
        
        # Fetch the last 100 messages from the group using the resolved entity
        last_messages = await client.get_messages(group_entity, limit=100)

        # Concatenate the last 50 messages
        concatenated_messages = "\n".join([msg.text for msg in last_messages])

        # Send the concatenated message to OpenAI for processing
        messages = [
            {"role": "system", "content": "о главном и коротко переписать по пунктам, каждый новый пункт начинать с новой строки и в конце дать объективное мнение ."},
            {"role": "user", "content": concatenated_messages}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response_text = response['choices'][0]['message']['content']

        # Send the processed response to the target group
        await client.send_message(GROUP_ID, response_text)

    except errors.ChannelPrivateError:
        print("The bot is not a member of the group or the group is private.")
    except Exception as e:
        print(f"An error occurred: {e}")

async def periodic_task():
    while True:
        await process_last_messages()
        await asyncio.sleep(3600)  # Sleep for 60 minutes

if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(periodic_task())
