from os import environ
from time import time
from asyncio import Lock
import discord
from actions import Actions
from helpers import count_tokens
from openai_models import chat_completion_functions, chat_completion

class DiscordActions(Actions):

  def __init__(self, client:discord.Client, message:discord.Message) -> None:
    super().__init__({})
    self.client = client
    self.context:list[discord.Message] = []
    self.file_ids = []
    self.instructions[0]['content'] += f" Your name is {environ['DISCORD_BOT_NAME']} or {environ['DISCORD_BOT_ID']}."
    self.message = message
    self.content = message.content

  async def __post_handler__(self) -> None:
    self.channel = await self.client.fetch_channel(self.message.channel.id)
    self.context = [m async for m in self.channel.history()]
    if self.channel.type == discord.ChannelType.text:
      self.channel = await self.channel.create_thread(name=round(time()*1000), message=self.message)
    elif self.channel.type == discord.ChannelType.public_thread:
      self.context.append(self.message)
    if any(self.client.user.mentioned_in(msg) for msg in self.context):
      return await chat_completion_functions(await self.messages_from_context(max_tokens=12288), self.available_functions)

  async def messages_from_context(self, count=None, max_tokens=12288):
    if count:
      self.context = await self.message.for_channel(self.message['channel_id'], params={'per_page':count})
    msgs = []
    tokens = count_tokens(self.instructions)
    for msg_json in self.context:
      if msg_json.author.bot:
        role = 'assistant'
      else:
        role = 'user'
      msg_json = {'role':role, 'content':msg_json.content}
      msg_tokens = count_tokens(msg_json)
      new_tokens = tokens + msg_tokens
      if new_tokens > max_tokens:
        break
      msgs.append(msg_json)
      tokens = new_tokens
    msgs.reverse()
    return self.instructions+msgs

  async def stream_reply(self, msgs:list, model='gpt-3.5-turbo-16k', max_tokens:int=None):
    if self.message.reference:
      reply_to = self.message.reference.message_id
    else:
      reply_to = None
    message:discord.Message = None
    buffer = []
    content = ''
    chunks_processed = []
    start_time = time()
    async with Lock():
      async for chunk in chat_completion(msgs, model=model, max_tokens=max_tokens):
        if not chunk:
          continue
        buffer.append(chunk)
        if (time() - start_time) * 1000 >= 1337:
          joined_chunks = ''.join(buffer)
          content = ''.join(chunks_processed)+joined_chunks
          if message:
            message = await message.edit(content=content)
          elif reply_to:
            message = await self.channel.send(content=content, reference=discord.MessageReference(message_id=reply_to, channel_id=self.message.channel.id))
          else:
            message = await self.channel.send(content=content)
          chunks_processed.append(joined_chunks)
          buffer.clear()
          start_time = time()
      if buffer:
        content = ''.join(chunks_processed)+''.join(buffer)
        if message:
          message = await message.edit(content=content)
        elif reply_to:
          await self.message.channel.send(content=content, reference=discord.MessageReference(message_id=reply_to, channel_id=self.message.channel.id))
        else:
          await self.message.channel.send(content=content)
