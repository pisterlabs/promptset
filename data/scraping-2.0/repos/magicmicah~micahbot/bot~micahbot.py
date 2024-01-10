import discord
from discord.ext import commands
import logging
import re
import ai
import reactions
import settings
import user
import utils
import typing
logger = logging.getLogger(__name__)
logger.info("Starting up...")

registry = user.UserRegistry()

intents = discord.Intents.all()
intents.message_content = True
intents.members = True

help_command = commands.DefaultHelpCommand(show_parameter_descriptions=False)

bot = commands.Bot(
    command_prefix="/",
    description="Micahbot is a fully featured Discord bot that does the needful things.",
    intents=intents,
    help_command=help_command,
)


@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game(name="Hacking up some silliness."))
    await bot.tree.sync()
    logger.info("Micahbot is ready!")

@bot.hybrid_command(name="micahart",)
async def micahart(ctx, model: typing.Literal['openai', 'replicate'], prompt: str = None):
    """Return an image from Replicate AI based on the prompts you provide.

    Arguments:
        prompt (str, optional): The text prompt used to generate the image.

    Returns:
        The generated image.

    Example:
        /micahart "A beautiful sunset." "people"
        https://prompthero.com/stable-diffusion-prompts
    """
    logger.info("Micahart invoked by user: " + ctx.author.name)
    if not prompt:
        await ctx.send("Please provide a prompt to generate an image.")
        return

    msg = await ctx.send(f"Generating {prompt}...")
    if model == "openai":
        image = await ai.get_openai_image(prompt)
    elif model == "replicate":
        image = await ai.get_replicate_image(prompt)
    await msg.edit(content=f"Prompt: {prompt}\n {image}")

@bot.hybrid_command(name="micahroll",)
async def micahroll(ctx, dice: str = None):
    """Roll a dice or two. Ex. /micahroll 1d20

    Arguments:
        dice (str, optional): The dice to roll.

    Returns:
        The result of the dice roll.

    Example:
        /micahroll 1d20
    """
    logger.info("Micahroll invoked by user: " + ctx.author.name)
    if not dice:
        await ctx.send("Please provide a dice to roll. Example: 1d20")
        return

    # parse dice string - 1d20, 2d6, etc.
    pattern = r'(\d+)d(\d+)'  # 1d20, 2d6, etc.
    match = re.match(pattern, dice)
    if match is None:
        await ctx.send(f"Invalid dice: {dice}")
        return
    num_dice = int(match.group(1))
    num_sides = int(match.group(2))
    if num_dice < 1 or num_dice > 5:
        await ctx.send(f"Invalid number of dice: {num_dice}")
        return
    if num_sides < 1 or num_sides > 20:
        await ctx.send(f"Invalid number of sides: {num_sides}")
        return
    
    msg = await ctx.send(f"Rolling {dice}...")
    rolls = [utils.random_number(1, num_sides) for _ in range(num_dice)]
    result = sum(rolls)
    await msg.edit(content=f"Rolling {dice}...\nRolls: {rolls}\nResult: {result}")

@bot.event
async def on_member_join(member):
    registry.add_user(member.id, member.name)
    for channel in member.guild.channels:
        if str(channel) == "general":
            await channel.send(f"""Welcome to the server {member.mention}!""")

@bot.event
async def on_member_remove(member):
    registry.remove_user(member.id)
    for channel in member.guild.channels:
        if str(channel) == "general":
            await channel.send(f"""{member.mention} has left the server.""")

@bot.event
async def on_message(message):

    if message.author == bot.user:
        return

    if isinstance(message.channel, discord.DMChannel):
        logger.info("DM received from user: " + message.author.name)
        user_registered = registry.is_user_in_registry(message.author.id)
        if user_registered is False:
            registry.add_user(message.author.id, message.author.name)
        
        clean_message = message.clean_content.strip()

        registry.append_user_message(message.author.id, "user", clean_message)

        user_messages = registry.get_user_messages(message.author.id)

        async with message.author.typing():


            response = await ai.get_openai_chat_completion(
                        model="gpt-3.5-turbo",
                        messages=user_messages,
                        user=str(message.author.id))
            
            if(response is None):
                final_response = "I don't know what to say to that."
            else:
                final_response = response
                registry.append_user_message(message.author.id, "assistant", final_response)
                split_messages = utils.split_string(final_response)
                for split_message in split_messages:
                    await message.reply(split_message)

    elif isinstance(message.channel, discord.TextChannel):
        # Add reactions to messages
        if message.channel is not None:
            react_ids = reactions.get_react_id(message)
            if react_ids is not None:
                for react_id in react_ids:
                    await message.add_reaction(react_id)
        
        # Random chance to get a GIF reaction
        number = utils.random_number(1, 1000)
        if number == 1:
            nouns = ai.get_nouns(message.content)
            noun_string = " ".join(nouns)
            gif_react = reactions.get_gif_react(noun_string)
            logger.info(f"Rolled a 1. Getting a GIF based off: {noun_string}")
            if gif_react is not None:
                await message.reply(gif_react)    
        
        # Check for PC language in this guild only
        if message.guild.id == 700525223790772255:
            check_message = ai.check_pc_language(message)
            if check_message is not None:
                await message.reply(check_message)

        # If bot is mentioned, respond
        if bot.user in message.mentions:
            logger.info("TextChannel received from user: " + message.author.name)
            user_registered = registry.is_user_in_registry(message.author.id)
            if user_registered is False and message.author != bot.user:
                # async with message.channel.typing():
                #     await message.channel.send(f"Hello {message.author.mention}! By messaging me, you agree to abide by the terms and conditions from OpenAI (https://openai.com/policies/terms-of-use).")
                registry.add_user(message.author.id, message.author.name)
            
            if ("where is" in message.content.lower()):
                await message.reply("I don't know where anything is. I'm just a bot.")
                return
            else:
                clean_message = message.clean_content.replace(f"@MicahBot", "").strip()
                
                registry.append_user_message(message.author.id, "user", clean_message)

                user_messages = registry.get_user_messages(message.author.id)

                async with message.channel.typing():


                    response = await ai.get_openai_chat_completion(
                                    model="gpt-3.5-turbo",
                                    messages=user_messages,
                                    user=str(message.author.id))
                    
                    if(response is None):
                        final_response = "I don't know what to say to that."
                    else:
                        final_response = response
                        registry.append_user_message(message.author.id, "assistant", final_response)
                        split_messages = utils.split_string(final_response)                    
                        for split_message in split_messages:
                            await message.reply(split_message)
                

    await bot.process_commands(message)

bot.run(settings.DISCORD_BOT_TOKEN)
