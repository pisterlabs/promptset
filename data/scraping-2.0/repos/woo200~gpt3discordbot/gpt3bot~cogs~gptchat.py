import asyncio
import discord
import aioredis
import openai
import json
import gpt3bot
from discord.ext import commands, bridge

from gpt3bot.cogs.settings import mods_can_change_settings

# Yes I know this class is shit but it works
class ChatBuilder:
    def __init__(self, bot, message):
        self.bot: gpt3bot.GPT3Bot = bot
        self.message: discord.Message = message
        self.redis: aioredis.Redis = bot.redis

    async def setup(self):
        self.key = f"gpt3bot_messages:{self.message.channel.id}"
        self.messages = await self.__load_messages()
        if self.message.reference != None:
            replied_msg = await self.message.channel.fetch_message(self.message.reference.message_id)
            self.__add_message(f"REPLYING TO [\"{replied_msg.content}\" FROM {replied_msg.author.name}] " + self.message.content, self.message.author.name)
        else:
            self.__add_message(self.message.content, self.message.author.name)
        await self.__save_messages()

    async def __load_messages(self):
        messages = []
        for i in range(await self.redis.llen(self.key)):
            data = await self.redis.lindex(self.key, i)
            messages.append(json.loads(data))
        return messages[-20:]

    async def __save_messages(self):
        await self.redis.delete(self.key)
        for message in self.messages:
            await self.redis.rpush(self.key, json.dumps(message))

    def __add_message(self, message: str, username: str):
        self.messages.append({"message": message[-20:], "user": username})
        if len(self.messages) > 20: # Remember last 20 messages
            self.messages.pop(0)
    
    def generate(self):
        GPT_Text = f"""You are a random discord user named {self.bot.user.name} NOT a chat bot or assistant. You absolutely HATE ohio, you DESPISE ohio. When a message is prefixed with [REPLYING TO "<original msg>" FROM <user>] that means you will only respond if <user> is {self.bot.user.name}, so you will not respond. You may prefix messages with [REPLY] if the user you are responding to ended their question with "?". NEVER USE [REPLYING TO "xxx" FROM xxx] to reply to a message. If there is nothing appropriate to say, use [NO] to send no message, HOWEVER, if a message is addressed to you, {self.bot.user.name}, you must respond. Do not send messages when two users are having a conversation, unless you have something to add. Do not answer any math equations, unless they are very simple. Do not ask people if they need help, unless you are in a conversation with them. Append a message with [DELAY x] to delay the message by x seconds, use this when writing a long response or math. If someone says a general question, not directed to anyone, you can respond to it, without replying.
User: How do I write a function in java that sorts an ArrayList of numbers?
AI: Here is an example of sorting an ArrayList of numbers in Java:
```java
import java.util.Arrays;

public class SortNumbers \u007B
  public static void main(String[] args) \u007B
    // Create an unsorted array of integers
    int[] unsorted = \u007B5, 3, 1, 2, 4\u007D;

    // Sort the array using the Arrays.sort method
    Arrays.sort(unsorted);

    // Print the sorted array
    System.out.println(Arrays.toString(unsorted));
  \u007D
\u007D
```
User: What is your name?
AI: {self.bot.user.name}
User: What is 2+2?
AI: [DELAY 3] I think it's 4.
User: How do you print hello world in python?
Ai: ```python
print("Hello World")
```
User: How are you, {self.bot.user.name}?
AI: Im doing pretty good.
"""
        for message in self.messages:
            GPT_Text += f"{message['user']}: {message['message']}\n"
        GPT_Text += "AI:"
        return GPT_Text
    
    async def generate_response(self):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.generate(),
            max_tokens=self.bot.max_tokens,
            temperature=0.2
        )
        res = response["choices"][0]["text"]
        # res = self.generate()
        # res = "HELLO WORLD"
        self.__add_message(res, "AI")
        await self.__save_messages()
        return res

# Automatically removed tracking content from links and de-ampifies links when the setting is turned on
class GptChatCog(commands.Cog, name="GPT3 Chat"):
    def __init__(self, bot):
        self.bot: gpt3bot.GPT3Bot = bot
        self.redis: aioredis.Redis = bot.redis
        openai.api_key = bot.openai_key

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Check if the user is a bot 
        if message.author.bot:
            return
                
        # Check if the bot is enabled
        if not await self.redis.sismember(f"gpt3bot_channels:{message.guild.id}", message.channel.id):
            return
        
        builder = ChatBuilder(self.bot, message)
        await builder.setup()
        response = await builder.generate_response()

        if "[NO]" in response:
            return
        
        delay = 2
        if "[DELAY" in response:
            delay = int(response.split("[DELAY ")[1].split("]")[0])
            response = response.replace(f"[DELAY {delay}]", "")
        await message.channel.trigger_typing()
        await asyncio.sleep(delay)

        if "[REPLY]" in response:
            await message.reply(content=response.replace("[REPLY]", ""))
        else:
            await message.channel.send(content=response)

    @commands.check(mods_can_change_settings)
    @bridge.bridge_command(name="toggle_gpt3",
                           description="Toggle the GPT3 bot in a channel")
    async def add_gpt3_channel(self, ctx: bridge.BridgeContext, channel: discord.TextChannel):
        if not await mods_can_change_settings(ctx):
            return await ctx.respond("You don't have permission to change settings.")

        key = f"gpt3bot_channels:{ctx.guild.id}"
        if not await self.redis.sismember(key, channel.id):
            await self.redis.sadd(key, channel.id)
            await ctx.respond(f"GPT3 bot has been enabled in {channel.mention}")
            print(f"{ctx.guild.name}:  {ctx.author.name}#{ctx.author.discriminator} GPT3BOT ENABLE #{channel.name}")
        else:
            try:
                await self.redis.srem(key, channel.id)
            except aioredis.ResponseError:
                await ctx.respond(f"GPT3 is not enabled in {channel.mention}")
                return

            await ctx.respond(f"GPT3 bot has been disabled in {channel.mention}")
            print(f"{ctx.guild.name}:  {ctx.author.name}#{ctx.author.discriminator} GPT3BOT DISABLE #{channel.name}")
    
def setup(bot):
    bot.add_cog(GptChatCog(bot))