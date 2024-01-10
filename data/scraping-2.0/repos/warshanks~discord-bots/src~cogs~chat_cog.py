"""
This module imports the necessary libraries for a Discord bot cog that interacts
with the OpenAI API to generate responses using the GPT-4 model. The bot can
process messages in specific channels and reply with generated content.

Libraries:
    - re: Used for regular expressions to split text and match patterns.
    - openai: Provides an interface to interact with the OpenAI API.
    - discord: A library for creating Discord bots and interacting with the Discord API.
    - discord.ext.commands: Provides a framework for creating commands for the Discord bot.
    - config: A custom module that contains configuration settings, such as API tokens,
              organization ID, and channel IDs for the Discord bot.
"""

import re
import openai
import discord
from discord.ext import commands
from config import openai_token, openai_org, channel_ids, gpt4_channels, lilith_channel

# Set OpenAI API key and organization
openai.api_key = openai_token
openai.organization = openai_org

bot = commands.Bot(command_prefix="~", intents=discord.Intents.all())


async def generate_response(message, conversation_log, openai_model):
    """
    This asynchronous function generates a response
    from the OpenAI model based on the conversation history.

    Args:
        message (discord.Message): The message received from the Discord channel.
        conversation_log (list): A list of dictionaries containing the conversation history.
        openai_model (str): The identifier of the OpenAI model to use for generating the response.

    Returns:
        str: The generated response from the OpenAI model.

    Raises:
        Exception: If there's an error in generating a response from the OpenAI API.

    Usage:
        To use this function, pass a Discord message object,
        a list with the conversation log,
        and the OpenAI model identifier.
        The function will return a generated response based on the conversation history.
    """

    # Get the last 10 messages from the channel and reverse the order
    previous_messages = [msg async for msg in message.channel.history(limit=10)]
    previous_messages.reverse()

    # Iterate through the previous messages
    for previous_message in previous_messages:
        # Ignore any message that starts with '!'
        if not previous_message.content.startswith('!'):
            # Determine the role based on whether the
            # author of the message is a bot or not.
            # This lets the AI know which of the previous messages it sent
            # and which were sent by the user.
            role = 'assistant' if previous_message.author.bot else 'user'

            # Add log item to conversation_log
            conversation_log.append({
                'role': role,
                'content': previous_message.content
            })

    # Send the conversation log to OpenAI to generate a response
    try:
        response = await openai.ChatCompletion.acreate(
            model=openai_model,
            messages=conversation_log,
            max_tokens=1024,
        )
    except Exception as error_message:
        return error_message

    # Return the response content
    return response['choices'][0]['message']['content']


async def send_sectioned_response(message, response_content, max_length=2000):
    """
    This asynchronous function sends a response message
    in sections if it exceeds the specified maximum length.

    Args:
        message (discord.Message): The message received from the Discord channel.
        response_content (str): The response content to be sent as a message.
        max_length (int, optional): The maximum length of a message. Defaults to 2000.

    Usage:
        To use this function, pass a Discord message object,
        the response content to be sent, and the optional max_length.
        The function will send the response in sections
        if it exceeds the specified maximum length.
    """

    # Split the response_content into
    # sentences using regular expression
    # The regex pattern looks for sentence-ending punctuation
    # followed by a whitespace character
    sentences = re.split(r'(?<=[.!?])\s+', response_content)

    # Initialize an empty section
    section = ""

    # Iterate through the sentences
    for sentence in sentences:
        # If the current section plus the next sentence exceeds the max_length,
        # send the current section as a message and clear the section
        if len(section) + len(sentence) + 1 > max_length:
            await message.reply(section.strip())
            section = ""

        # Add the sentence to the section
        section += " " + sentence

    # If there's any content left in the section, send it as a message
    if section:
        await message.reply(section.strip())


# noinspection PyShadowingNames
async def kc_conversation(message, openai_model):
    """
    This asynchronous function initiates a conversation with the user as KC the secretary,
    using the specified OpenAI model, and sends the generated response in sections.

    Args:
        message (discord.Message): The message received from the Discord channel.
        openai_model (str): The identifier of the OpenAI model to use for generating the response.

    Usage:
        To use this function,
        pass a Discord message object and the OpenAI model identifier.
        The function will generate a response based on the conversation
        history and send it in sections if it exceeds the specified maximum length.
    """

    try:
        # Create a log of the user's message and the bots response
        # send the typing animation while the bot is thinking
        async with message.channel.typing():
            conversation_log = [{'role': 'system', 'content':
                                 'You are a friendly secretary named KC.'}]

            response_content = await generate_response(message, conversation_log, openai_model)
            await send_sectioned_response(message, response_content)
    except Exception as error_message:
        await message.reply(f"Error: {error_message}")


# noinspection PyShadowingNames
class GPT4Cog(commands.Cog):
    """
    This class defines the GPT-4 Cog,
    which listens to messages in a specific channel and
    generates responses using the specified OpenAI model.

    Attributes:
        bot (discord.Client): The instance of the Discord bot.

    Methods:
        on_message(message): An event handler that is triggered when the bot receives a message.
        It checks if the message is valid and then generates a response using the
        kc_conversation function.

    Usage:
        To use this Cog, add it to the bot using the `add_cog()` method.
        The bot will then listen to messages in the specified channel and
        generate responses using the OpenAI model.
    """

    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Event handler for when the bot receives a message.
        This function checks if the message is valid
        and generates a response using the kc_conversation function.

        Args:
            message (discord.Message): The message received from the Discord channel.

        Usage:
            This function is used as a listener method within the GPT4Cog class.
            When the bot receives a message, it ignores messages from the bot itself,
            from channels other than the designated one,
            or that start with '!'.
            If the message is valid, the function generates a response using
            the kc_conversation function and the specified OpenAI model.
        """

        if (message.author.bot or
                message.author.system or
                message.channel.id not in gpt4_channels or
                message.content.startswith('!')):
            return

        # set the model to use
        openai_model = 'gpt-4'

        # Generate a response as KC
        await kc_conversation(message, openai_model)


# noinspection PyShadowingNames
class ChatCog(commands.Cog):
    """
        This class defines the GPT-3.5-Turbo chat Cog,
        which listens to messages in a specific channel and
        generates responses using the specified OpenAI model.

        Attributes:
            bot (discord.Client): The instance of the Discord bot.

        Methods:
            on_message(message): An event handler that is triggered when the bot receives a message.
            It checks if the message is valid and then generates a response using the
            kc_conversation function.

        Usage:
            To use this Cog, add it to the bot using the `add_cog()` method.
            The bot will then listen to messages in the specified channel and
            generate responses using the OpenAI model.
        """

    def __init__(self, bot):
        self.bot = bot

    # Event handler for when a message is sent in a channel
    @commands.Cog.listener()
    async def on_message(self, message):
        """
                Event handler for when the bot receives a message.
                This function checks if the message is valid
                and generates a response using the kc_conversation function.

                Args:
                    message (discord.Message): The message received from the Discord channel.

                Usage:
                    This function is used as a listener method within the GPT4Cog class.
                    When the bot receives a message, it ignores messages from the bot itself,
                    from channels other than the designated one,
                    or that start with '!'.
                    If the message is valid, the function generates a response using
                    the kc_conversation function and the specified OpenAI model.
                """
        if (message.author.bot or
                message.author.system or
                message.channel.id not in channel_ids or
                message.content.startswith('!')):
            return

        # set the model to use
        openai_model = 'gpt-3.5-turbo'

        # Generate a response as KC
        await kc_conversation(message, openai_model)

    @bot.tree.command(name='hype', description='Generate hype emojipasta')
    async def hype(self, ctx, about: str):
        """
        This function generates hype emojipasta based on the user input,
        using the specified OpenAI model.

        Args:
            ctx (commands.Context): The context in which the command is called.
            about (str): The topic for which to generate hype emojipasta.

        Usage:
            This function is used as a command method within a Discord bot class.
            When the 'hype' command is invoked,
            the bot will generate hype emojipasta based on the user's input,
            using the specified OpenAI model.
            If the generated response is too long,
            the bot will notify the user to try again.
        """

        # Defer the response to let the user know that the bot is working on the request
        await ctx.response.defer(thinking=True, ephemeral=False)
        conversation_log = [{'role': 'system', 'content': 'Generate really hype emojipasta about'},
                            {'role': 'user', 'content': about}]
        # Print information about the user, guild and channel where the command was invoked
        openai_model = 'gpt-3.5-turbo'
        try:
            # Generate a response using OpenAI API from the prompt provided by the user
            response = await openai.ChatCompletion.acreate(
                model=openai_model,
                messages=conversation_log,
                frequency_penalty=2.0,
                max_tokens=1024,
            )
            await ctx.followup.send(response['choices'][0]['message']['content'])
        except discord.errors.HTTPException:
            await ctx.followup.send("I have too much to say, please try again.")


# noinspection PyShadowingNames
class LilithCog(commands.Cog):
    """
    This class defines the Lilith Cog,
    which listens to messages in a specific channel
    and generates responses using the specified OpenAI model
    while role-playing as Lilith, daughter of Hatred, from the Diablo universe.

    Attributes:
        bot (discord.Client): The instance of the Discord bot.

    Methods:
        on_message(message): An event handler that is triggered when the bot receives a message.
        It checks if the message is valid and then generates a response using the generate_response
        function.

    Usage:
        To use this Cog, add it to the bot using the `add_cog()` method.
        The bot will then listen to messages in the specified channel and
        generate responses as Lilith using the OpenAI model.
    """

    def __init__(self, bot):
        self.bot = bot

    # Event handler for when the bot receives a message
    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Event handler for when the bot receives a message.
        This function checks if the message is valid and
        generates a response while role-playing as Lilith,
        daughter of Hatred, from the Diablo universe,
        using the generate_response function.

        Args:
            message (discord.Message): The message received from the Discord channel.

        Usage:
            This function is used as a listener method within the LilithCog class.
            When the bot receives a message, it ignores messages from the bot itself,
            from channels other than the designated one,
            or that start with '!'.
            If the message is valid, the function generates a response as Lilith
            using the generate_response function and the specified OpenAI model.
        """

        if (message.author.bot or
                message.author.system or
                message.channel.id != lilith_channel or
                message.content.startswith('!')):
            return
        openai_model = 'gpt-3.5-turbo'
        try:
            # Create a log of the user's message and the bot's response
            async with message.channel.typing():
                conversation_log = [{'role': 'system',
                                     'content':
                                         'Roleplay as Lilith, daughter of Hatred, '
                                         'from the Diablo universe.'}]

                response_content = await generate_response(message, conversation_log, openai_model)
                await message.reply(response_content)
        except Exception as error_message:
            await message.reply(f"Error: {error_message}")
