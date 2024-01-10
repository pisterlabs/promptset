import asyncio
import openai
import discord
import datetime

from discord.ext import commands


class BasicListeners(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    listening = None

    @commands.Cog.listener()
    async def on_message(self, ctx):
        if str(ctx.content) == "hey" and not ctx.author.bot and not self.bot.gpt_timeout:
            self.bot.gpt_timeout = True
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=1.0,
                    messages=[
                        {"role": "system", "content": "You are a character simulation that simulates a specified character."},
                        {"role": "system", "content": "This is an online chat between you and another user."},
                        {"role": "system", "content": "Act as if you are a Monty Python character."},
                        {"role": "system", "content": "Reply with an insulting greeting based on the other person's name."},
                        {"role": "system", "content": f"Name: {ctx.author.display_name}"},
                        {"role": "user", "content": "hey"}
                    ]
                )
            except openai.error.RateLimitError:
                print("Ratelimited by OpenAI!")
                return

            completion_content = completion.choices[0].message.content
            await ctx.reply(completion_content)
            await asyncio.sleep(2)
            self.bot.gpt_timeout = False

        if self.listening == ctx.author.id:
            file = open("notes.txt", "a")
            file.write(f"{ctx.content}\n\n")
            file.close()
            self.listening = None
            async with ctx.channel.typing():
                sleep_until = datetime.datetime.now() + datetime.timedelta(seconds=2)
                await discord.utils.sleep_until(sleep_until)
                await ctx.channel.send("alright i got you")

        if ctx.content == "DAVE listen up" and not ctx.author.bot:
            self.listening = ctx.author.id


async def setup(bot):
    print("Loading listeners extension..")
    await bot.add_cog(BasicListeners(bot))
