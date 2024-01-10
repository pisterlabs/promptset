import asyncio
import re

import discord
import openai
from discord.ext import commands

from . import data_management

#########################################

# Database Operations and Values

SETTINGS_TABLE_NAME = "openai_settings"
SETTINGS_COLUMNS = ("guild_id", "default_prompt", "key_list", "openai_status")


async def fetch_default_prompt(guild_id: int):
    return await data_management.fetch_entry(SETTINGS_TABLE_NAME, SETTINGS_COLUMNS[1], guild_id=guild_id,
                                             default_type=str)


async def fetch_key_list(guild_id: int):
    return await data_management.fetch_entry(SETTINGS_TABLE_NAME, SETTINGS_COLUMNS[2], guild_id=guild_id,
                                             default_type=list)


async def fetch_status(guild_id: int):
    return await data_management.fetch_entry(SETTINGS_TABLE_NAME, SETTINGS_COLUMNS[3], guild_id=guild_id)


async def write_default_prompt(guild_id: int, default_prompt: str):
    await data_management.update_entry(SETTINGS_TABLE_NAME, SETTINGS_COLUMNS[1], default_prompt, guild_id=guild_id)


async def write_key_list(guild_id: int, key_list: list):
    await data_management.update_entry(SETTINGS_TABLE_NAME, SETTINGS_COLUMNS[2], key_list, guild_id=guild_id)


async def write_status(guild_id: int, status: bool):
    await data_management.update_entry(SETTINGS_TABLE_NAME, SETTINGS_COLUMNS[3], status, guild_id=guild_id)


#########################################

NO_API_KEYS_MESSAGE = """There seem to be no API keys available.
- You can add a new API key to the bot using the command `/add_openai_key`
- You can get an OpenAI key by signing up for a free trial on https://beta.openai.com/.
To sign up you need a phone number, but you can use any free SMS receiver for sign up."""


def generate_message(message: str, api_key: str, user_name: str, default_prompt: str = None, history: str = None):
    openai.api_key = api_key
    prompt_parts = [f'{user_name}: """{message}"""', f'Bot: """']
    if history:
        prompt_parts.insert(0, history)
    elif default_prompt:
        prompt_parts.insert(0, default_prompt)
    prompt = "\n".join(prompt_parts)
    suffix = f'"""\n{user_name}: """'
    print(f"OpenAI Full Prompt:\n\n {prompt}")
    completion = openai.Completion.create(engine="text-davinci-003",
                                          prompt=prompt,
                                          max_tokens=400,
                                          stop='"""',
                                          suffix=suffix,
                                          user=user_name)
    reply = completion['choices'][0]['text']
    reply = reply.rstrip('"')
    print(f"OpenAI Full Completion & Suffix: \n\n {reply + suffix}")
    reply = re.sub(r"https?://.*\.\w{2,3}", "<snip>", reply)
    new_history = f'{prompt}{reply}"""'

    return reply, new_history


async def generate_message_async(message: str, api_key, user_name: str, default_prompt: str = None,
                                 history: str = None) -> tuple:
    loop = asyncio.get_running_loop()
    if history:
        history = "\n".join([default_prompt, history])
    reply, new_history = await loop.run_in_executor(None, generate_message, message, api_key, user_name, default_prompt,
                                                    history)
    return reply, new_history


async def reply_to_message(message: discord.Message, history: str = None):
    current_status = await fetch_status(message.guild.id)
    if not current_status:
        return

    api_keys = await fetch_key_list(message.guild.id)
    try:
        api_key = api_keys[0]
    except IndexError:
        await message.reply(NO_API_KEYS_MESSAGE)
        return

    cleaned_user_name = "".join([symbol for symbol in message.author.display_name if symbol.isalnum()])
    if not cleaned_user_name:
        cleaned_user_name = "Human"

    default_prompt = await fetch_default_prompt(message.guild.id)

    try:
        reply, new_history = await generate_message_async(message.clean_content, api_key, cleaned_user_name,
                                                          default_prompt, history)
        try:
            await message.reply(reply,
                                allowed_mentions=discord.AllowedMentions.none())
        except discord.errors.HTTPException:
            await message.reply("What?")

    except (openai.error.ServiceUnavailableError, openai.error.RateLimitError, openai.error.APIConnectionError):
        await message.reply("`OpenAI seems to be unavailable right now. Please try again.`")
    except openai.error.AuthenticationError:
        await message.reply("`Key has reached API limit. Try again to see if more keys are available.`")
        await delete_current_api_key(message.guild)


#########################################

class ChangePromptModal(discord.ui.Modal):
    def __init__(self, bot):
        super().__init__(title="Set the new OpenAI prompt.")
        self.bot = bot

    async def on_submit(self, interaction: discord.Interaction):
        await write_default_prompt(interaction.guild_id, self.children[0].value)
        await interaction.response.send_message("Updated the OpenAI prompt.")


async def delete_current_api_key(guild: discord.Guild):
    key_list = await fetch_key_list(guild.id)
    del key_list[0]
    await write_key_list(guild.id, key_list)


#########################################


class GPTInteraction(commands.Cog):

    def __init__(self, bot):
        self.bot = bot

    async def cog_load(self):
        await data_management.create_table(SETTINGS_TABLE_NAME, SETTINGS_COLUMNS)

    @discord.app_commands.command(
        name="_edit_openai_prompt",
        description="Change the prompt for the OpenAI bot.")
    @discord.app_commands.guild_only()
    @discord.app_commands.default_permissions(administrator=True)
    async def edit_openai_prompt(self, interaction: discord.Interaction):
        current_prompt = await fetch_default_prompt(interaction.guild_id)
        change_prompt_modal = ChangePromptModal(self.bot)
        change_prompt_modal.add_item(
            discord.ui.TextInput(label='New Prompt:',
                                 style=discord.TextStyle.paragraph,
                                 default=current_prompt,
                                 min_length=0,
                                 max_length=None))
        await interaction.response.send_modal(change_prompt_modal)

    @discord.app_commands.command(
        name="add_openai_key",
        description="Add an OpenAI key for the bot to use.")
    @discord.app_commands.describe(openai_key="OpenAI API key.")
    async def add_openai_key(self, interaction: discord.Interaction, openai_key: str):
        try:
            reply, new_history = await generate_message_async("How are you doing?", openai_key, "TestUser")
        except openai.error.RateLimitError:
            await interaction.response.send_message(
                "The key either has reached the quota limit or we are being rate limited.", ephemeral=True)
            return
        except openai.error.AuthenticationError:
            await interaction.response.send_message("This is an invalid key.", ephemeral=True)
            return
        if reply:
            key_list = await fetch_key_list(interaction.guild_id)
            key_list.append(openai_key)
            key_list = list(set(key_list))
            await write_key_list(interaction.guild_id, key_list)
            await interaction.response.send_message(f"{interaction.user.mention} Key has been added!",
                                                    ephemeral=True)
            await interaction.channel.send("New OpenAI API key has been added.")

    @discord.app_commands.command(
        name="_toggle_openai",
        description="Enable/Disable the OpenAI bot.")
    @discord.app_commands.guild_only()
    @discord.app_commands.default_permissions(administrator=True)
    async def toggle_openai(self, interaction: discord.Interaction):
        current_status = await fetch_status(interaction.guild_id)
        if current_status:
            current_status = False
            await write_status(interaction.guild_id, current_status)
            await interaction.response.send_message("Deactivated the OpenAI interaction.")
        else:
            current_status = True
            await write_status(interaction.guild_id, current_status)
            await interaction.response.send_message("Activated the OpenAI interaction.")

    async def get_message_history(self, message: discord.Message, history_strings: object = None) -> list:
        """
        Recursively fetches message history from cached messages and constructs a message history in acorrdance with
        the prompt.
        """
        if not history_strings:
            history_strings = list()
        if message.author == self.bot.user:
            history_strings.append(f'Bot: """{message.clean_content}"""')
        else:
            cleaned_user_name = "".join([symbol for symbol in message.author.display_name if symbol.isalnum()])
            if not cleaned_user_name:
                cleaned_user_name = "Human"
            history_strings.append(f'{cleaned_user_name}: """{message.clean_content}"""')

        if len(history_strings) > 8:
            return history_strings

        try:
            history_strings = await self.get_message_history(message.reference.cached_message,
                                                             history_strings=history_strings)
        except AttributeError:
            pass
        return history_strings

    @commands.Cog.listener(name="on_message")
    async def openai_reply(self, message: discord.Message):
        """
        Reply to @mentions or when the 'reply' interface is used.
        """

        if message.author == self.bot.user:
            return
        if not message.guild:
            return

        mention = f"<@{self.bot.user.id}>"
        if mention in message.content:
            await reply_to_message(message)
            return

        try:
            replied_to_user = message.reference.cached_message.author
            if replied_to_user == self.bot.user:
                message_history = await self.get_message_history(message.reference.cached_message)
                message_history.reverse()
                new_history = "\n".join(message_history)
                await reply_to_message(message, new_history)
                return
        except AttributeError:
            return


async def setup(bot):
    await bot.add_cog(GPTInteraction(bot))
