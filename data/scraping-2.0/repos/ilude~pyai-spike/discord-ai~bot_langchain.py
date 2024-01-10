import discord
import responses
import json, codecs
from langchain.docstore.document import Document

from dotenv import dotenv_values, find_dotenv

config = dotenv_values(find_dotenv())

async def send_message(message, user_message, is_private):
  try:
    response = responses.get_response(user_message)
    await message.author.send(response) if is_private else await message.channel.send(response)
    
  except Exception as e:
    print(e)
    
def run_discord_bot():
  intents = discord.Intents.default()
  intents.message_content = True
  client = discord.Client(intents=intents)

  @client.event
  async def on_ready():
    print(f'{client.user} is now running!')
    messages = []
    for server in client.guilds:
      for channel in server.channels:
        if str(channel.type) == 'text' and 'general' in  channel.name:
          print(channel)
          #text_channel_list.append(channel)
          filename = str(channel.name)
          async for msg in channel.history(limit=None, oldest_first=True):
            if msg.content:
              doc = {
                'content': msg.content,
                'author': msg.author.name,
                'created_at': msg.created_at,
                'jump_url': msg.jump_url
              }
              messages.append(doc)
    
    with open('general.json', 'wb') as f:
      json.dump(messages, codecs.getwriter('utf-8')(f), ensure_ascii=False, default = str, indent=2)     
    print("completed export")

    
  @client.event
  async def on_message(message):
    if message.author == client.user:
      return
    
    username = str(message.author)
    user_message = str(message.content)
    channel = str(message.channel)
    
    print(f'{username} said: "{user_message}" ({channel})')
    
    if user_message[0] == '?':
      user_message = user_message[1:]
      await send_message(message, user_message, is_private=(True))
    else:
      await send_message(message, user_message, is_private=(False))

  client.run(config['DISCORD_BOT_TOKEN'])  