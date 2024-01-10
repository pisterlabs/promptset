import time
from math import floor

from gpt_j.Basic_api import simple_completion

from database_utilities import get_guild_permissions
from discord_utilities import send_fancy_message
import openai

def add_commands(bot):

    cooldowns = {}
    cooldown = 8

    def check_content(content):

        response = openai.Completion.create(
            engine="content-filter-alpha",
            prompt="<|endoftext|>" + content + "\n--\nLabel:",
            temperature=0,
            max_tokens=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=10
        )

        output = int(response.choices[0].text)

        return output < 1

    @bot.command(name='continue', help='Trys to continue the text using gpt-j', aliases=['c'], usage='<text> [t=<temperature>] [p=top_p]')
    async def completion(ctx, *text):

        if not text:
            await ctx.channel.send('Please provide some text to continue.')
            return

        final_text = ""
        temperature = 0.5
        top_p = 0.9

        for word in text:

            if word.startswith('t='):
                temperature = float(word[2:])
            elif word.startswith('p='):
                top_p = float(word[2:])
            else:
                final_text = final_text + word + " "

        final_text = final_text.strip()

        completion = ""

        try:
            completion = simple_completion(prompt=final_text, length=64, temp=top_p, top=temperature)

        except Exception as e:
            await send_fancy_message(ctx, "Error: " + str(e), color=0xaa8888)
            return

        await send_fancy_message(ctx, completion) # Temp and top_p are flipped

    @bot.command(name='davincicontinue', help='Trys to continue the text using the davinci engine', aliases=['dc'],
                 usage='<text> [t=<temperature>] [p=<top_p>] [engine=<engine>]')
    async def davinci_completion(ctx, *text):

        # Check permissions to see if this guild can use this command
        permissions = get_guild_permissions(ctx.guild.id)
        if "davinci_completion" not in permissions:
            await send_fancy_message(ctx, "You do not have permission to use this command.", color=0xaa8888)
            return

        if not text:
            await ctx.channel.send('Please provide some text to continue.')
            return

        if ctx.guild.id in cooldowns:
            if time.time() - cooldowns[ctx.guild.id] < cooldown:
                await send_fancy_message(ctx, "Please wait " + str(floor(cooldown - (time.time() - cooldowns[ctx.guild.id]))) + " seconds before using this command again.", color=0xaa8888)
                return
            else:
                cooldowns[ctx.guild.id] = time.time()

        else:

            cooldowns[ctx.guild.id] = time.time()

        final_text = ""
        temperature = 0.5
        top_p = 0.9
        engine = "text-davinci-001"

        for word in text:

            if word.startswith('t='):
                temperature = float(word[2:])
            elif word.startswith('p='):
                top_p = float(word[2:])
            elif word.startswith('engine='):
                engine = word[7:]
            else:
                final_text = final_text + word + " "

        final_text = final_text.strip()

        if not check_content(final_text):
            await send_fancy_message(ctx, "Please provide some text that is not controversial, or offensive.",
                                     color=0xaa8888)
            return

        completion = ""

        try:

            completion = final_text + openai.Completion.create(
                prompt=final_text,
                max_tokens=64,
                temperature=temperature,
                top_p=top_p,
                engine=engine,
            ).choices[0].text

        except Exception as e:

            await send_fancy_message(ctx, "Error: " + str(e), color=0xaa8888)
            return

        await send_fancy_message(ctx, completion)

