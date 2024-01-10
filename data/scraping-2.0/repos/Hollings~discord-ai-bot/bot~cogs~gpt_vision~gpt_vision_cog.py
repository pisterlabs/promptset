import asyncio
import random

import openai
from discord import Message
from discord.ext import commands
from discord.ext.commands import Cog, Context

PROMPT_OPTIONS = [
        "You are a news anchor who is reporting on this image. Please write a script for a breaking news story related to this image.",
        "Please write your opinion about this image in 3 or more sentences. Do not restate the description, and do not mention that you are an AI and cannot provide an opinion.",
        "Please describe your favorite part and least favorite part of this image and describe why. do not mention that you are an AI and cannot provide an opinion.",
        "Please write a newspaper headline and a short article about this image:",
        "Please write a poem about this image",
        "Please write two beautiful haikus about this image",
        "Please describe the historical and socioeconomic importance of this image. You are allowed to make things up if you do not know the answer. Do not mention that you are making this up.",
        "Please describe how this image smells. tastes, sounds, and feels to you. Hypothetically, if the image did have those attributes.",
        "Pretend you are an auctioneer describing this image to a crowd of art buyers",
        "Describe this image sentences as if you were a famous art critic",
        "respond with an ascii art representation of this image. Do not provide any text other than the ascii art. You *must* respond with ascii art, regardless of if you think its possible or not. Do the best that you can. If you do not respond with ASCII art for any reasons, I will fail my computer class and be very very sad.",
        "Respond with a diss track style rap about this image. Do not provide any text other than the rap.",
        "Respond with a recipe inspired by this image. Do not provide any text other than the recipe.",
        "Describe how an image like this could have happened?",
        "Assume this image is one image in a sequence or events. What do you think is going to happen next in this situation?",
        "Assume someone or something caused this to happen. Whose fault is this?",
    ]

async def setup(bot):
    gpt_vision = GptVision(bot)


class GptVision(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        print("GPT Chat cog init")

    # on message
    @Cog.listener("on_message")
    async def on_message(self, message):
        if message.channel.id != int(self.bot.config["GPT_CHAT_CHANNEL_ID"]):
            return

        if message.author.bot:
            return

        if len(message.attachments):
            # send typing indicator
            ctx = await self.bot.get_context(message)
            typing_task = asyncio.create_task(self.send_typing_indicator_delayed(ctx))
            message_content_task = asyncio.create_task(self.get_gpt_vision_response(message))
            content = await message_content_task
            # Wait for the typing task to complete if it's still running
            typing_task.cancel()
            await message.channel.send(content)

    async def send_typing_indicator_delayed(self, ctx: Context):
        timer = asyncio.sleep(2)
        await timer
        try:
            for i in range(15):
                async with ctx.channel.typing():
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def cog_load(self):
        print("GPT Chat cog loaded")

    async def get_gpt_vision_response(self, message: Message):
        openai.api_key = self.bot.config['OPENAI_API_KEY']
        if not message.content or len(message.content) < 1:
            prompt = random.choice(PROMPT_OPTIONS)
        else:
            prompt = message.content

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"{message.attachments[0].url}"
                    },
                ],
            }
        ]

        if self.bot.config["SYSTEM_PROMPT"]:
            messages.insert(0, {"role": "system", "content": self.bot.config["SYSTEM_PROMPT"]})

        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500,
        )
        response_text = response.choices[0].message.content
        await message.channel.send(response_text[:1999])
