"""
RinBot v2.2.0
made by rin
"""

# Imports
import os, sys, platform, subprocess, asyncio, discord, random, pytz
from discord import app_commands
from discord.ext import commands, tasks
from discord.ext.commands import Bot, Context
from rinbot.base import db_manager
from rinbot.base.exceptions import Exceptions as E
from rinbot.base.logger import logger
from rinbot.base.helpers import load_lang, format_exception, strtobool, format_date
from rinbot.base.colors import *
from rinbot.fortnite.daily_shop import get_shop
from dotenv import load_dotenv
from datetime import datetime

# Load verbose
text = load_lang()

# Make sure cache dirs exist
try:
    folders = ["cache", "cache/fun", "cache/chatlog", "cache/stablediffusion", "cache/ocr",
               "rinbot/assets/images/fortnite/downloaded", "rinbot/assets/images/fortnite/images"]
    for folder in folders:
        folder_path = f"{os.path.realpath(os.path.dirname(__file__))}/{folder}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"{text['INIT_CREATED_DIRECTORY']} '{folder}'")
except Exception as e:
    e = format_exception(e)
    sys.exit(f"{text['INIT_CREATED_DIRECTORY_ERROR']} {e}")

# Check if ffmpeg is present
try:
    if platform.system() == "Windows":
        if not os.path.isfile(f"{os.path.realpath(os.path.dirname(__file__))}/rinbot/bin/ffmpeg.exe"):
            sys.exit(f"{text['INIT_FFMPEG_NOT_FOUND_WINDOWS']}")
    elif platform.system() == "Linux":
        try:
            result = subprocess.run(["ffmpeg", "-version"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            if result.returncode != 0:
                sys.exit(f"{text['INIT_FFMPEG_NOT_FOUND_LINUX']}")
        except FileNotFoundError:
            sys.exit(f"{text['INIT_FFMPEG_NOT_FOUND_LINUX']}")
except Exception as e:
    e = format_exception(e)
    sys.exit(f"{text['INIT_FFMPEG_CHECK_ERROR']} {e}")

# Load env vars
load_dotenv()
RINBOT_VER = os.getenv('RINBOT_VER')
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
BOT_PREFIX = os.getenv('DISCORD_BOT_PREFIX')
BOT_ACTIVITY_CHANGE_INTERVAL = os.getenv('DISCORD_BOT_ACTIVITY_CHANGE_INTERVAL')
BOORU_ENABLED = strtobool(os.getenv('BOORU_ENABLED'))
RULE34_ENABLED = strtobool(os.getenv('RULE34_ENABLED'))
AI_ENABLED = strtobool(os.getenv('AI_ENABLED'))
AI_CHAR_NAME = os.getenv('AI_CHAR_NAME')
AI_ENDPOINT_KOBOLD = os.getenv('AI_ENDPOINT_KOBOLD')
AI_CHANNEL = os.getenv('AI_CHANNEL')
AI_CHAT_HISTORY_LINE_LIMIT = os.getenv('AI_CHAT_HISTORY_LINE_LIMIT')
AI_MAX_NEW_TOKENS = os.getenv('AI_MAX_NEW_TOKENS')
AI_LANGUAGE = os.getenv('AI_LANGUAGE')
FN_SHOP_ENABLED = strtobool(os.getenv('FNBR_DAILY_SHOP_ENABLED'))
FN_UPDATE_TIME = os.getenv('FNBR_UPDATE_TIME')

# Bot intents (many)
intents = discord.Intents.all()
intents.dm_messages = True
intents.dm_reactions = True
intents.dm_typing = True
intents.guild_messages = True
intents.guild_reactions = True
intents.guild_scheduled_events = True
intents.guild_typing = True
intents.guilds = True
intents.integrations = True
intents.invites = True
intents.voice_states = True
intents.webhooks = True
intents.members = True
intents.message_content = True
intents.moderation = True
intents.presences = True
intents.emojis_and_stickers = True
intents.messages = True
intents.emojis = True
intents.reactions = True
intents.typing = True
intents.bans = True

# Create bot object and parse it to the database manager
bot = Bot(
    command_prefix=commands.when_mentioned_or(BOT_PREFIX),
    intents = intents, help_command=None,)
bot.logger = logger
db_manager.declare_bot(bot)

# Vars and consts
freshstart = True
message_count = {}
time_window_milliseconds = 5000
max_msg_per_window = 5
author_msg_times = {}
utc = pytz.utc
HOURS = {text['HOURS']}
MINUTES = {text['MINUTES']}
SECONDS = {text['SECONDS']}
OWNER = {text['OWNER']}
ADMIN = {text['ADMIN']}

# Will I use AI?
if AI_ENABLED:
    from langchain.llms.koboldai import KoboldApiLLM
    
    # Specific AI settings
    bot.endpoint = str(AI_ENDPOINT_KOBOLD)
    if len(bot.endpoint.split("/api")) > 0:
        bot.endpoint = bot.endpoint.split("/api")[0]
    bot.chatlog_dir = "cache/chatlog"
    bot.endpoint_connected = False
    bot.channel_id = AI_CHANNEL
    bot.num_lines_to_keep = int(AI_CHAT_HISTORY_LINE_LIMIT)
    bot.guild_ids = [int(x) for x in AI_CHANNEL.split(",")]
    bot.debug = True
    bot.char_name = AI_CHAR_NAME
    bot.endpoint_type = "Kobold"
    bot.llm = KoboldApiLLM(endpoint=bot.endpoint, max_length=AI_MAX_NEW_TOKENS)

# Show fortnite daily shop
async def show_fn_daily_shop():
    # Grab shop data
    bot.logger.info(f"{text['INIT_FN_UPDATING']}")
    shop = await get_shop()
    img_dir = f"{os.path.realpath(os.path.dirname(__file__))}/rinbot/assets/images/fortnite/images"
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    embed = discord.Embed(
        title=f"{text['INIT_DAILY_SHOP_EMBED'][0]} **{format_date(shop['date'])}**",
        description=f"{text['INIT_DAILY_SHOP_EMBED'][1]} **{len(img_files)}** {text['INIT_DAILY_SHOP_EMBED'][2]} **{shop['count']}** {text['INIT_DAILY_SHOP_EMBED'][3]}",
        color=PURPLE)

    # Show store for each guild that has it enabled
    for guild in bot.guilds:
        check = await db_manager.get_daily_shop_channel(guild.id)
        if check:
            if check[0] == 1:
                channel = await bot.fetch_channel(check[1])
                await channel.send(embed=embed)
                batches = []
                for i in range(0, len(img_files), 6):
                    batch = []
                    img_names = img_files[i:i+6]
                    for img in img_names:
                        img_path = os.path.join(img_dir, img)
                        batch.append(discord.File(img_path, filename=img))
                    batches.append(batch)
                for batch in batches:
                    await channel.send(files=batch)

    # Delete images to prevent cache buildup and mixing with other shops
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        os.remove(file_path)
    bot.logger.info(f"{text['INIT_FN_UPDATED']}")
async def daily_shop_scheduler():
    while True and FN_SHOP_ENABLED:
        time = FN_UPDATE_TIME.split(":")
        curr_time = datetime.now(utc).strftime("%H:%M:%S")
        target_time = f"{time[0]}:{time[1]}:{time[2]}"
        if curr_time == target_time:
            await show_fn_daily_shop()
        await asyncio.sleep(1)

# Change status every 5 minutes (if there are more than 1)
@tasks.loop(minutes=int(BOT_ACTIVITY_CHANGE_INTERVAL))
async def status_loop():
    chosen = random.choice(text['INIT_ACTIVITY'])
    await bot.change_presence(
        activity=discord.CustomActivity(name=chosen))

# When ready
@bot.event
async def on_ready() -> None:
    # Initial logger info (splash)
    bot.logger.info("--------------------------------------")
    bot.logger.info(f"  > {RINBOT_VER}")
    bot.logger.info("--------------------------------------")
    bot.logger.info(f" {text['INIT_SPLASH_LOGGED_AS']} {bot.user.name}")
    bot.logger.info(f" {text['INIT_SPLASH_API_VER']} {discord.__version__}")
    bot.logger.info(f" {text['INIT_SPLASH_PY_VER']} {platform.python_version()}")
    bot.logger.info(f" {text['INIT_SPLASH_RUNNING_ON']} {platform.system()}-{platform.release()} ({os.name})")
    bot.logger.info("--------------------------------------")
    
    # Check if all members are present in the economy
    bot.logger.info(f"{text['INIT_CHECKING_ECONOMY']}")
    for guild in bot.guilds:
        for member in guild.members:
            await db_manager.add_user_to_currency(member.id, guild.id)
        bot.logger.info(f"{text['INIT_CURR_GUILD'][0]} {guild.member_count} {text['INIT_CURR_GUILD'][1]} {guild.name}")

    # Make sure bot has a history of all joined guilds
    bot.logger.info(f"{text['INIT_CHECKING_GUILDS']}")
    joined = await db_manager.get_joined_ids()
    guilds = []
    for i in joined: guilds.append(i[0])
    for guild in bot.guilds:
        if str(guild.id) not in guilds:
            await db_manager.add_joined_on(guild.id)

    # Start tasks
    status_loop.start()
    asyncio.create_task(daily_shop_scheduler())

    # Sync slash commands
    bot.logger.info(f"{text['INIT_SYNCHING_COMMANDS']}")
    await bot.tree.sync()
    
# Member welcome
@bot.event
async def on_member_join(member:discord.Member):
    # Add new member to the economy
    await db_manager.add_user_to_currency(int(member.id), int(member.guild.id))
    
    # Check if guild has a welcome channel setup and show welcome message
    try:
        guild_data = await db_manager.get_welcome_channel(member.guild.id)
        if guild_data:
            if guild_data[0] == 1:
                channel = bot.get_channel(int(guild_data[1]))
                if channel:
                    embed = discord.Embed(
                        title=f"{text['INIT_NEW_MEMBER_TITLE'][0]} {member.mention}{text['INIT_NEW_MEMBER_TITLE'][1]}",
                        description=f"{guild_data[2]}", color=YELLOW)
                    await channel.send(embed=embed)
    except Exception as e:
        e = format_exception(e)
        bot.logger.error(f"{text['INIT_ERROR_BASE']} {e}")

# Save new guild ID's when joining
@bot.event
async def on_guild_join(guild:discord.Guild):
    bot.logger.info(f"{text['INIT_JOINED_GUILD']} '{guild.name}'! ID: {guild.id}")
    joined_on = await db_manager.get_joined_ids()
    for i in joined_on:
        if not guild.id == int(i[0]):
            await db_manager.add_joined_on(guild.id)
    
    # Add all members to the economy
    uids = [i.id for i in guild.members]
    for id in uids:
        await db_manager.add_user_to_currency(id, guild.id)

# Remove guild ID's when leaving
@bot.event
async def on_guild_remove(guild:discord.Guild):
    remove = await db_manager.remove_joined_on(guild.id)
    if remove:
        bot.logger.info(f"{text['INIT_LEFT_GUILD']} '{guild.name}'! ID: {guild.id}")
    else:
        bot.logger.error(f"{text['INIT_LEFT_GUILD']} '{guild.name}'! ID: {guild.id}, {text['INIT_LEFT_GUILD_DELETE_ERROR']}")

# Process messages
@bot.event
async def on_message(message:discord.Message):
    try:
        # Do not interact with self or other bots
        if message.author == bot.user or message.author.bot:
            return
        
        # Anti-spam measure for economy system
        global author_msg_times
        aid = message.author.id
        ct = datetime.now().timestamp() * 1000
        if not author_msg_times.get(aid, False):
            author_msg_times[aid] = []
        author_msg_times[aid].append(ct)
        et = ct - time_window_milliseconds
        em = [mt for mt in author_msg_times[aid] if mt < et]
        for mt in em:
            author_msg_times[aid].remove(mt)
        if not len(author_msg_times[aid]) > max_msg_per_window:
            await db_manager.update_message_count(message.author.id, message.guild.id)
        
        # Process commands
        await bot.process_commands(message)
        
    except AttributeError:
        pass

# Show executed commands on the log
@bot.event
async def on_command_completion(ctx:Context) -> None:
    comm_name = ctx.command.qualified_name.split(" ")
    comm = str(comm_name[0])
    bot.logger.info(f"{text['INIT_COMM_EXECUTED_GUILD'][0]} {comm} {text['INIT_COMM_EXECUTED_GUILD'][1]} (ID: {ctx.guild.id}) {text['INIT_COMM_EXECUTED_GUILD'][2]} {ctx.author} (ID: {ctx.author.id})"
                    if ctx.guild is not None else
                    f"{text['INIT_COMM_EXECUTED_DMS'][0]} {comm} {text['INIT_COMM_EXECUTED_DMS'][1]} {ctx.author} (ID: {ctx.author.id}) {text['INIT_COMM_EXECUTED_DMS'][2]}")

# What to do when commands go nuh-uh
@bot.event
async def on_command_error(ctx:Context, error) -> None:
    if isinstance(error, commands.CommandOnCooldown):
        minutes, seconds = divmod(error.retry_after, 60)
        hours, minutes = divmod(minutes, 60)
        hours = hours % 24
        embed = discord.Embed(
            description=f"{text['INIT_COMM_ERROR_DELAY']} {f'{round(hours)} {HOURS}' if round(hours) > 0 else ''} {f'{round(minutes)} {MINUTES}' if round(minutes) > 0 else ''} {f'{round(seconds)} {SECONDS}' if round(seconds) > 0 else ''}.",
            color=0xE02B2B,)
        await ctx.send(embed=embed)
    elif isinstance(error, E.UserBlacklisted):
        embed = discord.Embed(
            description=f"{text['INIT_BLOCKED']}", color=0xE02B2B)
        await ctx.send(embed=embed)
        if ctx.guild:
            bot.logger.warning(
                f"{ctx.author} (ID: {ctx.author.id}) {text['INIT_TRIED_COMM']} {text['INIT_TRIED_COMM_GUILD']} {ctx.guild.name} (ID: {ctx.guild.id}), {text['INIT_TRIED_COMM_BLOCKED']}")
        else:
            bot.logger.warning(
                f"{ctx.author} (ID: {ctx.author.id}) {text['INIT_TRIED_COMM']} {text['INIT_TRIED_ON_DMS']}, {text['INIT_TRIED_COMM_BLOCKED']}")
    elif isinstance(error, E.UserNotOwner):
        embed = discord.Embed(
            description=f"{text['INIT_NOT_OWNER']}", color=0xE02B2B)
        await ctx.send(embed=embed)
        if ctx.guild:
            bot.logger.warning(
                f"{ctx.author} (ID: {ctx.author.id}) {text['INIT_TRIED_COMM_CLASS']} `{OWNER}` {ctx.guild.name} (ID: {ctx.guild.id}), {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
        else:
            bot.logger.warning(
                f"{ctx.author} (ID: {ctx.author.id}) {text['INIT_TRIED_COMM_CLASS']} `{OWNER}` {text['INIT_TRIED_ON_DMS']}, {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
    elif isinstance(error, E.UserNotAdmin):
        embed = discord.Embed(
            description=f"{text['INIT_NOT_ADMIN']}", color=0xE02B2B)
        await ctx.send(embed=embed)
        if ctx.guild:
            bot.logger.warning(
                f"{ctx.author} (ID: {ctx.author.id}) {text['INIT_TRIED_COMM_CLASS']} `{ADMIN}` {ctx.guild.name} (ID: {ctx.guild.id}), {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
        else:
            bot.logger.warning(
                f"{ctx.author} (ID: {ctx.author.id}) {text['INIT_TRIED_COMM_CLASS']} `{ADMIN}` {text['INIT_TRIED_ON_DMS']}, {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
    elif isinstance(error, commands.MissingPermissions):
        embed = discord.Embed(
            description=f"{text['INIT_USER_NO_PERMS_1']}"
            + ", ".join(error.missing_permissions)
            + f"{text['INIT_USER_NO_PERMS_2']}",
            color=0xE02B2B,)
        await ctx.send(embed=embed)
    elif isinstance(error, commands.BotMissingPermissions):
        embed = discord.Embed(
            description=f"{text['INIT_BOT_NO_PERMS_1']}"
            + ", ".join(error.missing_permissions)
            + f"{text['INIT_BOT_NO_PERMS_2']}",
            color=0xE02B2B,)
        await ctx.send(embed=embed)
    elif isinstance(error, commands.MissingRequiredArgument):
        embed = discord.Embed(
            title=f"{text['ERROR']}",
            description=str(error).capitalize(),
            color=0xE02B2B,)
        await ctx.send(embed=embed)
    else:
        raise error
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error) -> None:
    if isinstance(error, app_commands.CommandOnCooldown):
        minutes, seconds = divmod(error.retry_after, 60)
        hours, minutes = divmod(minutes, 60)
        hours = hours % 24
        embed = discord.Embed(
            description=f"{text['INIT_COMM_ERROR_DELAY']} {f'{round(hours)} {HOURS}' if round(hours) > 0 else ''} {f'{round(minutes)} {MINUTES}' if round(minutes) > 0 else ''} {f'{round(seconds)} {SECONDS}' if round(seconds) > 0 else ''}.",
            color=0xE02B2B,)
        await interaction.response.send_message(embed=embed)
    elif isinstance(error, E.AC_UserBlacklisted):
        embed = discord.Embed(
            description=f"{text['INIT_BLOCKED']}", color=0xE02B2B)
        await interaction.response.send_message(embed=embed)
        if interaction.guild:
            bot.logger.warning(
                f"{interaction.user} (ID: {interaction.user.id}) {text['INIT_TRIED_COMM']} {text['INIT_TRIED_COMM_GUILD']} {interaction.guild.name} (ID: {interaction.guild.id}), {text['INIT_TRIED_COMM_BLOCKED']}")
        else:
            bot.logger.warning(
                f"{interaction.user} (ID: {interaction.user.id}) {text['INIT_TRIED_COMM']} {text['INIT_TRIED_ON_DMS']}, {text['INIT_TRIED_COMM_BLOCKED']}")
    elif isinstance(error, E.AC_UserNotOwner):
        embed = discord.Embed(
            description=f"{text['INIT_NOT_OWNER']}", color=0xE02B2B)
        await interaction.response.send_message(embed=embed)
        if interaction.guild:
            bot.logger.warning(
                f"{interaction.user} (ID: {interaction.user.id}) {text['INIT_TRIED_COMM_CLASS']} `{OWNER}` {interaction.guild.name} (ID: {interaction.guild.id}), {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
        else:
            bot.logger.warning(
                f"{interaction.user} (ID: {interaction.user.id}) {text['INIT_TRIED_COMM_CLASS']} `{OWNER}` {text['INIT_TRIED_ON_DMS']}, {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
    elif isinstance(error, E.AC_UserNotAdmin):
        embed = discord.Embed(
            description=f"{text['INIT_NOT_ADMIN']}", color=0xE02B2B)
        await interaction.response.send_message(embed=embed)
        if interaction.guild:
            bot.logger.warning(
                f"{interaction.user} (ID: {interaction.user.id}) {text['INIT_TRIED_COMM_CLASS']} `{ADMIN}` {interaction.guild.name} (ID: {interaction.guild.id}), {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
        else:
            bot.logger.warning(
                f"{interaction.user} (ID: {interaction.user.id}) {text['INIT_TRIED_COMM_CLASS']} `{ADMIN}` {text['INIT_TRIED_ON_DMS']}, {text['INIT_TRIED_COMM_NOT_IN_CLASS']}")
    elif isinstance(error, app_commands.MissingPermissions):
        embed = discord.Embed(
            description=f"{text['INIT_USER_NO_PERMS_1']}"
            + ", ".join(error.missing_permissions)
            + f"{text['INIT_USER_NO_PERMS_2']}",
            color=0xE02B2B,)
        await interaction.response.send_message(embed=embed)
    elif isinstance(error, app_commands.BotMissingPermissions):
        embed = discord.Embed(
            description=f"{text['INIT_BOT_NO_PERMS_1']}"
            + ", ".join(error.missing_permissions)
            + f"{text['INIT_BOT_NO_PERMS_2']}",
            color=0xE02B2B,)
        await interaction.response.send_message(embed=embed)
    else:
        raise error

# Loads extensions (command cogs)
async def load_extensions():
    if AI_ENABLED:
        for file in os.listdir(f"{os.path.realpath(os.path.dirname(__file__))}/rinbot/kobold/cogs"):
            if file.endswith(".py"):
                extension = file[:-3]
                await load_extension(extension, True)
    booru_ext = ["booru"]
    rule34_ext = ["rule34"]
    sum = booru_ext + rule34_ext
    for file in os.listdir(f"{os.path.realpath(os.path.dirname(__file__))}/rinbot/extensions"):
        if file.endswith(".py"):
            extension = file[:-3]
            if BOORU_ENABLED and extension in booru_ext:
                await load_extension(extension)
            elif RULE34_ENABLED and extension in rule34_ext:
                await load_extension(extension)
            is_general = all(extension not in sl for sl in sum)
            if is_general:
                await load_extension(extension)

# Loads an extension
async def load_extension(ext, ai=False):
    if not ai:
        try:
            await bot.load_extension(f"rinbot.extensions.{ext}")
            bot.logger.info(f"{text['INIT_EXTENSION_LOADED']} '{ext}'")
        except Exception as e:
            e = format_exception(e)
            bot.logger.error(f"{text['INIT_ERROR_LOADING_EXTENSION']} '{ext}': {e}")
    else:
        try:
            await bot.load_extension(f"rinbot.kobold.cogs.{ext}")
            if ext == 'languagemodel':
                bot.endpoint_connected = True
            bot.logger.info(f"{text['INIT_EXTENSION_LOADED']} '{ext}'")
        except Exception as e:
            e = format_exception(e)
            bot.logger.error(f"{text['INIT_ERROR_LOADING_EXTENSION']} '{ext}': {e}")

# RUN
asyncio.run(db_manager.check_owners())
asyncio.run(load_extensions())
try: bot.run(BOT_TOKEN)
except discord.errors.LoginFailure: sys.exit(f"{text['INIT_INVALID_TOKEN']}")