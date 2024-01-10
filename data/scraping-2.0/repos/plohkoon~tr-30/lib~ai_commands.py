from discord.app_commands import command, guilds
from discord import Interaction, Embed, Colour, ChannelType, Message, Thread
from discord.ext.commands import Cog, Bot
from .constants import TR_GUILD, IABW_GUILD, IABW_LOGO
from openai import Completion, Image, InvalidRequestError
import sys
from faker import Faker
from .chat_gpt import ChatGpt


# TODO: Update this to chat bot when I can get it working
DEFAULT_MODEL = "text-davinci-003"

MODEL_LIST = [
    "text-davinci-003",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
]

MAX_TOKENS = {
    "text-davinci-003": 4000,
    "text-curie-001": 2000,
    "text-babbage-001": 2000,
    "text-ada-001": 2000,
}

IMAGE_SIZES = {
    "small": "256x256",
    "medium": "512x512",
    "large": "1024x1024",
}


class AICommands(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot

        self.threads = {}

    @command(name="prompt", description="Prompt with the AI")
    @guilds(TR_GUILD, IABW_GUILD)
    async def prompt(self, interaction: Interaction, prompt: str, model: str = DEFAULT_MODEL) -> None:
        if model not in MODEL_LIST:
            embed = Embed(
                title="Error",
                color=Colour.red(),
                description="Something went wrong. Please try again later."
            )

            embed.add_field(
                name="Message",
                value="Model not found. To see available models user /list_models",
                inline=False
            )

            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        await interaction.response.defer()

        print(
            f"Generating a completion for the prompt: {prompt}", file=sys.stderr)

        embed = None

        try:
            response = Completion.create(
                model=model, prompt=prompt, max_tokens=MAX_TOKENS[model])

            message = response["choices"][0]["text"]

            embed = Embed(color=Colour.green())

            embed.set_thumbnail(url=IABW_LOGO)

            embed.set_author(name=interaction.user.display_name,
                             icon_url=interaction.user.avatar.url)

            embed.add_field(
                name="Prompt", value=prompt.capitalize(), inline=False)
            embed.add_field(name="Model", value=model, inline=False)
            embed.add_field(name="Response", value=message, inline=False)

            embed.set_footer(text="Powered by OpenAI")
        except InvalidRequestError as e:
            embed = Embed(
                title="Error",
                color=Colour.red(),
                description="Something went wrong. Please try again later."
            )

            embed.add_field(name="Message", value=e._message, inline=False)
        except Exception as e:
            embed = Embed(
                title="Error",
                color=Colour.red(),
                description="An unknown error occurred. Please try again later."
            )

            embed.add_field(name="message", value=str(e))
        finally:
            await interaction.followup.send(embed=embed)

    @command(name="list_models", description="List the available models")
    @guilds(TR_GUILD, IABW_GUILD)
    async def list_models(self, interaction: Interaction) -> None:
        print("Listing models for", interaction.user.display_name, file=sys.stderr)
        embed = Embed(
            title="Available Models",
            color=Colour.green(),
            description="There is a wide variety of models available."
        )

        embed.add_field(
            name="Default Model",
            inline=False,
            value=f"`{DEFAULT_MODEL}`",
        )

        embed.add_field(
            name="Davinci",
            inline=False,
            value="""
Key: `text-davinci-003`

Davinci is the most capable model family and can perform any task the other models can perform and often with less instruction. For applications requiring a lot of understanding of the content, like summarization for a specific audience and creative content generation, Davinci is going to produce the best results. These increased capabilities require more compute resources, so Davinci costs more per API call and is not as fast as the other models.

Good at: Complex intent, cause and effect, summarization for audience
"""
        )

        embed.add_field(
            name="Curie",
            inline=False,
            value="""
Key: `text-curie-001`

Curie is extremely powerful, yet very fast. While Davinci is stronger when it comes to analyzing complicated text, Curie is quite capable for many nuanced tasks like sentiment classification and summarization. Curie is also quite good at answering questions and performing Q&A and as a general service chatbot.

Good at: Language translation, complex classification, text sentiment, summarization
"""
        )

        embed.add_field(
            name="Babbage",
            inline=False,
            value="""
Key: `text-babbage-001`

Babbage can perform straightforward tasks like simple classification. It’s also quite capable when it comes to Semantic Search ranking how well documents match up with search queries.

Good at: Moderate classification, semantic search classification
"""
        )

        embed.add_field(
            name="Ada",
            inline=False,
            value="""
Key: `text-ada-001`

Ada is usually the fastest model and can perform tasks like parsing text, address correction and certain kinds of classification tasks that don’t require too much nuance. Ada’s performance can often be improved by providing more context.

Good at: Parsing text, simple classification, address correction, keywords
"""
        )

        embed.set_footer(
            text="Powered by OpenAI. For more information see https://beta.openai.com/docs/models/finding-the-right-model")

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @command(name="visualize", description="Make an image")
    @guilds(TR_GUILD, IABW_GUILD)
    async def visualize(self, interaction: Interaction, prompt: str, size: str = "medium") -> None:
        if size not in IMAGE_SIZES:
            embed = Embed(
                title="Error",
                color=Colour.red(),
                description="Something went wrong. Please try again later."
            )

            embed.add_field(
                name="Message",
                value="Model not found. Invalid image size. Use one of: small, medium, large",
                inline=False
            )

            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        await interaction.response.defer()

        print(
            f"Generating an image according to the prompt: {prompt}", file=sys.stderr)

        embed = None

        try:
            response = Image.create(
                prompt=prompt, size=IMAGE_SIZES[size], n=1, response_format="url", user=f"{interaction.user.id}")
            embed = Embed(color=Colour.green())

            embed.set_thumbnail(url=IABW_LOGO)

            embed.set_author(name=interaction.user.display_name,
                             icon_url=interaction.user.avatar.url)

            embed.add_field(
                name="Prompt", value=prompt.capitalize(), inline=False)
            embed.add_field(name="Size", value=size.capitalize(), inline=False)

            embed.set_image(url=response["data"][0]["url"])

            embed.set_footer(text="Powered by OpenAI")
        except InvalidRequestError as e:
            embed = Embed(
                title="Error",
                color=Colour.red(),
                description="Something went wrong. Please try again later."
            )

            embed.add_field(name="Message", value=e._message, inline=False)
        except Exception as e:
            embed = Embed(
                title="Error",
                color=Colour.red(),
                description="An unknown error occurred. Please try again later."
            )

            embed.add_field(name="message", value=str(e))
        finally:
            await interaction.followup.send(embed=embed)

    @command(name="chat", description="Start a chat thread")
    @guilds(TR_GUILD, IABW_GUILD)
    async def chat(self, interaction: Interaction, thread_name: str = None) -> None:
        if interaction.channel.type != ChannelType.text:
            await interaction.response.send_message(
                "This command can only be used in text channels",
                ephemeral=True
            )
            return

        if thread_name is None:
            fake = Faker()
            thread_name = fake.name()
            while thread_name in self.threads:
                thread_name = fake.name()

        await interaction.response.defer()

        chatter = ChatGpt()

        # TODO make sure that chatter initializes correctly

        thread = await interaction.channel.create_thread(
            name=thread_name,
            type=ChannelType.public_thread,
        )

        self.threads[thread.id] = chatter

        await thread.join()
        await thread.add_user(interaction.user)

        await thread.send("Hello! I'm a chatbot. Ask me anything!")

        await interaction.followup.send(
            "Chat thread created",
            ephemeral=True
        )

    @Cog.listener()
    async def on_message(self, message: Message) -> None:
        if message.author.bot:
            return

        thread = message.channel
        id = thread.id

        if thread.type != ChannelType.public_thread or thread.owner.id != self.bot.user.id or id not in self.threads.keys():
            return

        chatter = self.threads[id]

        try:
            async with message.channel.typing():
                response = None
                for segment in chatter.chat(message.content):
                    if response is None:
                        response = await message.channel.send(segment)
                    else:
                        response = await response.edit(content=segment)
        except Exception as e:
            print(e)
            await message.channel.send("Something went wrong. Please try again later.")
