import os
import pathlib
import shutil
import sys
import appdirs


template = """from discord import app_commands
from discord.ext import commands
from discord.ext.commands import Context

import openai
import re

def replace_emoji(emoji_name: str, emoji_map):
    if emoji_name in emoji_map:
        if emoji_map[emoji_name][1]:
            return f"<a:{{emoji_name}}:{{emoji_map[emoji_name][0]}}>"
        else:
            return f"<:{{emoji_name}}:{{emoji_map[emoji_name][0]}}>"
    elif emoji_name.upper() in emoji_map:
        if emoji_map[emoji_name.upper()][1]:
            return f"<a:{{emoji_name.upper()}}:{{emoji_map[emoji_name.upper()][0]}}>"
        else:
            return f"<:{{emoji_name.upper()}}:{{emoji_map[emoji_name.upper()][0]}}>"
    elif emoji_name.lower() in emoji_map:
        if emoji_map[emoji_name.lower()][1]:
            return f"<a:{{emoji_name.lower()}}:{{emoji_map[emoji_name.lower()][0]}}>"
        else:
            return f"<:{{emoji_name.lower()}}:{{emoji_map[emoji_name.lower()][0]}}>"
    else:
        return f":{{emoji_name}}:"


class {class_name}(commands.Cog, name="{command_name}"):
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_command(
        name="{command_name}",
        description="Generate a completion for {command_name}",
    )
    @app_commands.describe(
        prompt="The prompt to pass to your model: Default=\\"\\"",
        temp="What sampling temperature to use. Higher values means more risks: Min=0 Max=1 Default={temp_default}",
        presence_penalty="Number between -2.0 and 2.0. Positive values will encourage new topics: Min=-2 Max=2 Default={pres_default}",
        frequency_penalty="Number between -2.0 and 2.0. Positive values will encourage new words: Min=-2 Max=2 Default={freq_default}",
        max_tokens="The max number of tokens to generate. Each token costs credits: Default={max_tokens_default}",
        stop="Whether to stop after the first sentence: Default={stop_default}",
        bold="Whether to bolden the original prompt: Default={bold_default}")
    async def {command_name}(self, context: Context, prompt: str = "", temp: float = {temp_default},
                       presence_penalty: float = {pres_default}, frequency_penalty: float = {freq_default}, max_tokens: int = {max_tokens_default},
                       stop: bool = {stop_default}, bold: bool = {bold_default}):
        temp = min(max(temp, 0), 1)
        presPen = min(max(presence_penalty, -2), 2)
        freqPen = min(max(frequency_penalty, -2), 2)

        await context.defer()
        try:
            openai.api_key = "{openai_key}"
            response = openai.Completion.create(
                engine="{model_id}",
                prompt=prompt,
                temperature=temp,
                frequency_penalty=presPen,
                presence_penalty=freqPen,
                max_tokens=max_tokens,
                echo=False,
                stop='.' if stop else None,
            )
            emojied_response = re.sub(r":(\w+):", lambda match: replace_emoji(
                match.group(1), context.bot.emoji_map), f"{{'**' if bold and prompt else ''}}{{prompt}}{{'**' if bold and prompt else ''}}{{response[\'choices\'][0][\'text\']}}")
            await context.send(emojied_response[:2000])
        except Exception as error:
            print({error})
            await context.send(
                {error}
            )


async def setup(bot):
    await bot.add_cog({class_name}(bot))
"""

config_dir = pathlib.Path(appdirs.user_data_dir(appname="discordai"))


def gen_new_command(model_id: str, command_name: str, temp_default: float, pres_default: float, freq_default: float,
                    max_tokens_default: int, stop_default: bool, openai_key: str, bold_default: bool):
    if getattr(sys, 'frozen', False):
        # The code is being run as a frozen executable
        data_dir = pathlib.Path(appdirs.user_data_dir(appname="discordai"))
        cogs_path = data_dir / "discordai" / "bot" / "cogs"
        if not os.path.exists(cogs_path):
            data_dir = pathlib.Path(sys._MEIPASS)
            og_cogs_path = data_dir / "discordai" / "bot" / "cogs"
            shutil.copytree(og_cogs_path, cogs_path)
    else:
        # The code is being run normally
        template_dir = pathlib.Path(os.path.dirname(__file__))
        cogs_path = template_dir / "bot"/ "cogs"
    with open(pathlib.Path(cogs_path, f"{command_name}.py"), "w") as f:
        os.makedirs(cogs_path, exist_ok=True)
        f.write(
            template.format(
                model_id=model_id, class_name=command_name.capitalize(),
                command_name=command_name, temp_default=float(temp_default),
                pres_default=float(pres_default),
                freq_default=float(freq_default),
                max_tokens_default=max_tokens_default, stop_default=stop_default, openai_key=openai_key, bold_default = bold_default,
                error="f\"Failed to generate valid response for prompt: {prompt}\\nError: {error}\""))
        print(f"Successfully created new slash command: /{command_name} using model {model_id}")


def delete_command(command_name: str):
    confirm = input("Are you sure you want to delete this command? This action is not reversable. Y/N: ")
    if confirm not in ["Y", "y", "yes", "Yes", "YES"]:
        print("Cancelling command deletion...")
        return
    if getattr(sys, 'frozen', False):
        # The code is being run as a frozen executable
        data_dir = pathlib.Path(appdirs.user_data_dir(appname="discordai"))
        cogs_path = data_dir / "discordai" / "bot" / "cogs"
        if not os.path.exists(cogs_path):
            data_dir = pathlib.Path(sys._MEIPASS)
            og_cogs_path = data_dir / "discordai" / "bot" / "cogs"
            shutil.copytree(og_cogs_path, cogs_path)
    else:
        # The code is being run normally
        template_dir = pathlib.Path(os.path.dirname(__file__))
        cogs_path = template_dir / "bot" / "cogs"
    try:
        os.remove(pathlib.Path(cogs_path, f"{command_name}.py"))
        print(f"Successfully deleted command: /{command_name}")
    except FileNotFoundError:
        print("Failed to delete command: No command with that name was found.")
