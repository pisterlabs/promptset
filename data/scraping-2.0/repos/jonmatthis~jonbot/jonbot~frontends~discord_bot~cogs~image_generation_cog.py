import discord
from discord import Forbidden
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from jonbot import logger
from jonbot.backend.ai.image_generation.image_generator import ImageGenerator


class ImageGeneratorCog(discord.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.image_generator = ImageGenerator()

    @discord.slash_command(name="image", help="Generates an image based on the given query string")
    async def generate_image(self, ctx, query: str):
        message = await ctx.send(f"Generating image for `{query}`....")
        # Use the ImageGenerator class to generate, download, and save the image
        image_filename = await self.image_generator.generate_image(query)

        # Return a message with the query and the image as an attachment
        await message.edit(content=f"Image for \n\n ```\n\n{query}\n\n```\n\n", file=discord.File(image_filename))

    @discord.slash_command(name="dream_this_chat",
                           help="summarize this chat and generate an image based on the summary")
    @discord.option(
        name="temperature",
        description="the temperature to use for the image generation (higher temperature = more randomness)",
        input_type=float,
        required=False,
        default=.7
    )
    @discord.option(
        name="summary_prompt",
        description="the prompt that will beused to convert this chat into an image generation prompt",
        input_type=str,
        required=False,
        default="Use this text as a starting point to  generate a beautiful, complex, and detailed prompt for an image generation AI"
    )
    async def dream_this_chat(self,
                              ctx,
                              summary_prompt: str = "Use this text as a starting point to generate a beautiful, complex, and detailed prompt for an image generation AI",
                              temperature: float = .7,
                              ):
        response_message = await ctx.send(f"Generating image for chat in `{ctx.channel.name}`....")
        chat_string = await self._get_chat_string(ctx.channel)

        prompt_template = (summary_prompt +
                           "\n\n++++++++++"
                           "\n\n {text} "
                           "\n\n++++++++++"
                           "\n\n Remember, your job is to " + summary_prompt +
                           "\n\n Do not include any text in your response other than the summary"
                           "\n\n Keep your answer less than 1900 characters"
                           )

        llm = ChatOpenAI(temperature=temperature,
                         model_name="gpt-4-1106-preview")

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = await chain.ainvoke(input={"text": chat_string})

        # if len(response.content) > 1800:
        #     prompt_for_message = response.content[:1800] + "...\n (prompt truncated due to Discord's limit)"
        # else:
        #     prompt_for_message = response.content
        #
        # response_message_content = f"Image generation prompt: \n\n ```\n\n{prompt_for_message}\n\n```\n\n||\n\ngenerating image......"
        #
        # await response_message.edit(content=response_message_content)

        image_filename = await self.image_generator.generate_image(response.content)

        # response_message_content = response_message_content.split("||")[0]

        await response_message.edit(content="",
                                    file=discord.File(image_filename))

    async def _get_chat_string(self,
                               channel: discord.abc.Messageable,
                               list_length: int = 50
                               ) -> str:
        channel_messages = []
        try:
            logger.info(f"Scraping channel: {channel}")
            async for message in channel.history(limit=list_length, oldest_first=True):
                channel_messages.append(message)
            logger.info(f"Scraped {len(channel_messages)} messages from channel: {channel}")

        except Forbidden:
            logger.warning(f"Missing permissions to scrape channel: {channel}")

        chat_string = "\n\n".join([f"{message.content}" for message in channel_messages])
        return chat_string
