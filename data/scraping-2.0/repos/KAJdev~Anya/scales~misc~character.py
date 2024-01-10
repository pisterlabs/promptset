import asyncio
import random
import time
from naff import InteractionContext, listen, Extension, slash_command, slash_option
from os import getenv
import openai
import models

from naff.api.events import MessageCreate

NAMES = [
    'anya',
]

PROMPT = """
The following is a conversation with Anya who is a telepathic esper. Anya is a short young girl with fair skin and green eyes who is six years old. She is impressionable and loves spies.
Her father is named loid forger, a spy. and her mother is named yor forger, an assassin. She also has a dog named bond that can see the future.

{past}
{name}: {message}
anya:
"""

class Engine:
  value = "text-davinci-003"
client = openai.Client(getenv('OPENAI_TOKEN'), default_engine=Engine)

class LimitedList(list):
    """
    A list with a max amount of entries
    """

    def __init__(self, length: int):
        self.length = length

    def append(self, item):
        if len(self) >= self.length:
            self.pop(0)
        super().append(item)

class Interaction:
    """
    An interaction between someone and the bot
    """

    def __init__(self, name: str, message: str, response: str) -> None:
        self.name = name
        self.message = message
        self.response = response

    def __str__(self) -> str:
        return f"{self.name}: {self.message}\nAnya: {self.response}"

class Memory:
    """
    Anya's memory
    """

    def __init__(self, length: int = 5) -> None:
        self.data = {}
        self.length = length

    def get_past(self, user: int) -> str:
        memories = self.data.get(user, LimitedList(self.length))
        return "\n".join([str(m) for m in memories])

    def remember(self, user: int, interaction: Interaction) -> None:
        memories = self.data.get(user, LimitedList(self.length))
        
        for memory in memories:
            if interaction.response.lower() == memory.response.lower():
                return # don't add duplicate responses
        
        memories.append(interaction)
        self.data[user] = memories

class Character(Extension):

    def __init__(self, bot):
        super().__init__()
        self.memory = Memory()
        self.past_messages = LimitedList(100)

    @slash_command("train", description="train anya with a message")
    @slash_option("prompt", "what is said to anya",
        opt_type=3,
        required=True
    )
    @slash_option("completion", "what should be said by anya",
        opt_type=3,
        required=True,
    )
    async def train(self, ctx: InteractionContext, prompt: str, completion: str):
        """
        Train Anya with a message
        """

        intactn = {
            'prompt': prompt,
            'completion': completion,
        }
        
        await self.bot.db._update("training_data", {'prompt': prompt}, {'$set': intactn}, upsert=True)

        amount = await self.bot.db.db.training_data.count_documents({})

        await ctx.send(f"Inserted interaction `#{amount}`:\n\n```json\n{intactn}```")

    @slash_command("data", description="view the last few JSONL lines of training data")
    async def data(self, ctx: InteractionContext):
        last_few = await self.bot.db.db.training_data.find({}).sort('_id', -1).limit(10).to_list(length=10)

        await ctx.send(f"Last few interactions:\n\n```json\n{last_few}```")

    @listen()
    async def on_message_create(self, event: MessageCreate):
        message = event.message

        if message.author.bot or len(message.content) > 1000:
            return

        guild_stuff: models.Guild = await self.bot.db.fetch_guild(message.guild.id)

        if not guild_stuff.module_enabled(models.ModuleToggles.CHARACTER):
            return

        elif message.content.startswith("!memdump"):
            await message.reply(f"```{self.memory.data}```"[:1999])

        elif message.content.startswith("!forget"):
            if message.author.id in self.memory.data:
                del self.memory.data[message.author.id]

            await message.reply("I forgot everything about you.")

        elif message.content.startswith("!raw"):
            t = time.time()

            this_prompt = message.content[4:].strip("\n ")

            if len(this_prompt) > 1000:
                await message.reply("That prompt is too long.")
                return

            response = await client.complete(this_prompt, temperature=0.9, max_tokens=100, top_p=1, full_response=True)
            choice = response["choices"][0]
            text = choice["text"]#.strip("\n ")

            await message.reply(f"`Finish Reason`: {choice['finish_reason']} **|** `Latency`: {(time.time()-t)*1000:.0f}ms```ansi\n{this_prompt}\u001b[37m{text}```")


        # see if we should respond
        elif (
            f"<@{self.bot.user.id}>" in message.content or 
            f"<@!{self.bot.user.id}>" in message.content or 
            any(x.lower() in message.content.lower() for x in NAMES) or
            (message.message_reference and message.message_reference.message_id in self.past_messages)
        ):
            reading_Extension = 30
            
            # these are conditions where the bot should respond more promptly, so active conversations can exist
            if (
                (message.message_reference and message.message_reference.message_id in self.past_messages) or
                (f"<@{self.bot.user.id}>" in message.content or f"<@!{self.bot.user.id}>" in message.content) or
                (message.author.id in self.memory.data)
            ):
                reading_Extension = 3
            
            await asyncio.sleep(random.random() * reading_Extension) # some reaction time and reading time

            await message.channel.trigger_typing()

            content = message.content.replace(f"<@{self.bot.user.id}>", "").replace(f"<@!{self.bot.user.id}>", "").strip()
            self.bot.info("Message received: {}".format(content))

            this_prompt = PROMPT.format_map({'name': message.author.username, 'message': content, 'past': self.memory.get_past(message.author.id)})

            self.bot.info(f"Anya: {this_prompt}")

            response = await client.complete(this_prompt, temperature=0.9, max_tokens=64, top_p=1)
            response = response.strip("\n")
            
            first_length = len(response)
            response = response.split("\n")[0]
            
            if first_length > len(response):
                self.bot.error("cut off excess")

            response = response.replace("\n", " ")

            self.memory.remember(message.author.id, Interaction(message.author.username, content, response))
            
            typing_time = len(response) * 0.05
            while typing_time > 0:
                reduction = min(typing_time, 9)
                await message.channel.trigger_typing()
                await asyncio.sleep(reduction)
                typing_time -= reduction

            msg = await message.reply(response)
            self.past_messages.append(msg.id)
    
def setup(bot):
    Character(bot)