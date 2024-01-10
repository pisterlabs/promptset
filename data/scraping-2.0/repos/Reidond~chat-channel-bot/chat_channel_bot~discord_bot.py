import os
from .config import Config
import discord
from typing import Optional
from discord import app_commands
import traceback
import openai
import json
import asyncio

Config.init_config("./config.ini")

openai_api_key = Config.get_or_else("credentials", "OPENAI_API_KEY", "sk-xxxx")
os.environ["OPENAI_API_KEY"] = openai_api_key

discord_token = Config.get_or_else("credentials", "DISCORD_TOKEN", "token")
discord_guild_id = Config.get_or_else("credentials", "DISCORD_GUILD_ID", "0")


MY_GUILD = discord.Object(id=discord_guild_id)  # replace with your guild id


from .embedchain_async import App as EmbedChainApp, aadd, aadd_local, aquery


class MyClient(discord.Client):
    embedchain_chat_bot: Optional[EmbedChainApp | None] = None

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        # A CommandTree is a special type that holds all the application command
        # state required to make it work. This is a separate class because it
        # allows all the extra state to be opt-in.
        # Whenever you want to work with application commands, your tree is used
        # to store and work with them.
        # Note: When using commands.Bot instead of discord.Client, the bot will
        # maintain its own tree instead.
        self.tree = app_commands.CommandTree(self)
        self.embedchain_chat_bot = EmbedChainApp()

    # In this basic example, we just synchronize the app commands to one guild.
    # Instead of specifying a guild to every command, we copy over our global commands instead.
    # By doing so, we don't have to wait up to an hour until they are shown to the end-user.
    async def setup_hook(self):
        # This copies the global commands over to your guild.
        self.tree.copy_global_to(guild=MY_GUILD)
        await self.tree.sync(guild=MY_GUILD)


intents = discord.Intents.default()
client = MyClient(intents=intents)


@client.event
async def on_ready():
    if not client.user:
        return
    print(f"Logged in as {client.user} (ID: {client.user.id})")


class TrainModal(discord.ui.Modal, title="Train the bot"):
    train_data = discord.ui.TextInput(
        label="What do you want to train on?",
        style=discord.TextStyle.long,
        placeholder="Type your text...",
        required=False,
        max_length=300,
    )

    async def on_submit(self, interaction: discord.Interaction):
        async def train_model():
            systemMessage = """
            From the text below extract all occurences of youtube video url, pdf file url, web pages url.
            Include also text without those urls as one string.
            Only answer in json no other text is required.
            Json format:`{"text":"","youtube_videos":[""],"web_pages":[""],"pdfs":[""]}`
            """
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": systemMessage},
                    {"role": "user", "content": self.train_data.value},
                ],
                temperature=0,
            )
            completionRaw = response.choices[0].message.content
            completion = json.loads(completionRaw)

            try:
                for ytv in completion["youtube_videos"]:
                    await aadd(client.embedchain_chat_bot, "youtube_video", ytv)
                for wp in completion["web_pages"]:
                    await aadd(client.embedchain_chat_bot, "web_page", wp)
                for pdf in completion["pdfs"]:
                    await aadd(client.embedchain_chat_bot, "pdf_file", pdf)
                await aadd_local(client.embedchain_chat_bot, "text", completion["text"])
            except ValueError as e:
                print(e)

            await interaction.edit_original_response(content="Success training model.")

        asyncio.create_task(train_model())
        await interaction.response.send_message("Training model...", ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await interaction.response.send_message("Oops! Something went wrong.", ephemeral=True)
        # Make sure we know what the error actually is
        traceback.print_exception(type(error), error, error.__traceback__)


@client.tree.command(
    name="train",
    description="Train the bot with new data",
)
@app_commands.describe()
async def train(interaction: discord.Interaction):
    if client.embedchain_chat_bot is None:
        return
    await interaction.response.send_modal(TrainModal())


@client.tree.command(
    name="query",
    description="Query the bot with prompt",
)
@app_commands.describe(
    query="The prompt to query the bot with",
)
async def query(interaction: discord.Interaction, query: str):
    if client.embedchain_chat_bot is None:
        return

    async def query_model():
        answer = await aquery(client.embedchain_chat_bot, query)
        systemMessage = """
        From the text below find if the text is in negative, questionable form like:
        I don't know; Can't find any information;
        Only answer in json no other text is required.
        Set success field only if text is not in negative, questionable form.
        Json format:`{"success":false}`
        """
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": systemMessage},
                {"role": "user", "content": answer},
            ],
            temperature=0,
        )
        completionRaw = response.choices[0].message.content
        completion = json.loads(completionRaw)

        if not completion["success"]:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are helpfull assistant."},
                    {"role": "user", "content": query},
                ],
                temperature=0,
            )
            completionRaw = response.choices[0].message.content
            return await interaction.edit_original_response(content=completionRaw)

        await interaction.edit_original_response(content=answer)

    asyncio.create_task(query_model())
    await interaction.response.send_message("Querying model...")


client.run(discord_token)
