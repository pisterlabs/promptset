#!/usr/bin/env python3
import asyncio, base64, datetime, io, json, logging, os, random, time
import discord, openai
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from openai import AsyncOpenAI

###########
# Logging #
########### 
log = logging.getLogger('super-pal')
log.setLevel(logging.INFO)
log_handler = logging.FileHandler(filename='discord-super-pal.log', encoding='utf-8', mode='w')
dt_fmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter('[{asctime}] [{levelname:<8}] {name}: {message}', dt_fmt, style='{')
log_handler.setFormatter(formatter)
log.addHandler(log_handler)

##################
# Env. variables #
##################
load_dotenv()
TOKEN = os.environ['SUPERPAL_TOKEN']
GUILD_ID = int(os.environ['GUILD_ID'])
EMOJI_GUILD_ID = GUILD_ID if os.environ['EMOJI_GUILD_ID'] is None else int(os.environ['EMOJI_GUILD_ID'])
CHANNEL_ID = int(os.environ['CHANNEL_ID'])
ART_CHANNEL_ID = CHANNEL_ID if os.environ['ART_CHANNEL_ID'] is None else int(os.environ['ART_CHANNEL_ID'])
VOICE_CHANNELS = os.environ['VOICE_CHANNELS']
GPT_ASSISTANT_ID = os.environ['GPT_ASSISTANT_ID']
GPT_ASSISTANT_THREAD_ID = os.environ['GPT_ASSISTANT_THREAD_ID']

(base_reqnotmet,karatechop_reqnotmet,ai_reqnotmet) = (TOKEN is None or GUILD_ID is None or CHANNEL_ID is None, 
                                                      VOICE_CHANNELS is None,
                                                      os.environ['OPENAI_API_KEY'] is None)
RUNTIME_WARN_MSG = 'WARN: Super Pal will still run but you are very likely to encounter run-time errors.'
if base_reqnotmet:
    log.warn(f'Base requirements not fulfilled. Please provide TOKEN, GUILD_ID, CHANNEL_ID.\n{RUNTIME_WARN_MSG}\n')
if karatechop_reqnotmet:
    log.warn(f'Karate chop requirements not fulfilled. Please provide VOICE_CHANNELS.\n{RUNTIME_WARN_MSG}\n')
if ai_reqnotmet:
    log.warn(f'OpenAI requirements not fulfilled. Please provide api key.\n{RUNTIME_WARN_MSG}\n')

###################
# Message strings #
###################
COMMANDS_MSG = (f'**!spotw @name**\n\tPromote another user to super pal of the week. Be sure to @mention the user.\n'
    f'**!spinthewheel**\n\tSpin the wheel to choose a new super pal of the week.'
    f'**!cacaw**\n\tSpam the channel with party parrots.\n'
    f'**!meow**\n\tSpam the channel with party cats.\n'
    f'**!surprise** your text here\n\tReceive an AI-generated image in the channel based on the text prompt you provide.\n'
    f'**!karatechop**\n\tMove a random user to AFK voice channel.' )
GAMBLE_MSG = ( f'Respond to the two polly polls to participate in Super Pal of the Week Gambling™.\n'
    f'- Choose your challenger\n'
    f'- Make your wager\n\n'
    f'You will be given 100 points weekly so feel free to go all-in.\n\n'
    f'*The National Problem Gambling Helpline (1-800-522-4700) is available 24/7 and is 100% confidential.*' )
WELCOME_MSG = ( f'Welcome to the super pal channel.\n\n'
                f'Use super pal commands by posting commands in chat. Examples:\n'
                f'( !commands (for full list) | !surprise your text here | !karatechop | !spotw @name | !meow )' )
GPT_PROMPT_MSG = ( f'You are a helpful assistant named Super Pal Bot. '
                    f'You help the members of a small Discord community called Bringus. '
                    f'Each week a new super pal is chosen at random from the list of Bringus members.' )
# Define available tools for this assistant.
GPT_ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "is_member_super_pal",
            "description": "Check if the given member is currently super pal",
            "parameters": {
                "type": "object",
                "properties": {
                    "member": {
                        "type": "string",
                        "description": "The member name, e.g. clippy",
                    },
                },
                "required": ["member"],
            }
        }
    }
]

################
# OpenAI setup #
################
async def is_member_super_pal(member: str):
    guild = bot.get_guild(GUILD_ID)
    member = discord.utils.get(guild.members, name=member)
    super_pal_role = discord.utils.get(guild.roles, name='Super Pal of the Week')
    if super_pal_role in member.roles:
        return f"Yes, {member} is the super pal."
    else:
        return f"No, {member} is not the super pal."

async def respond_to_user(user_message: discord.Message):
    log.info(f"{user_message.author.name} said \"{user_message.content}\"")
    # Create OpenAI client and assistant.
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    try: # Try to get existing assistant.
        assistant = await client.beta.assistants.retrieve(
            assistant_id=GPT_ASSISTANT_ID
        )
    except openai.NotFoundError as e: # Assistant not found. We will create thread.
        log.warn(f"Assistant ID not found. Creating new Assistant.\nError: {e}")
        assistant = await client.beta.assistants.create(
            name="Super Pal Bot",
            instructions=GPT_PROMPT_MSG,
            tools=GPT_ASSISTANT_TOOLS,
            model="gpt-3.5-turbo-1106"
        )
    try: # Try to get existing thread.
        thread = await client.beta.threads.retrieve(
            thread_id=GPT_ASSISTANT_THREAD_ID
        )
    except openai.NotFoundError as e: # Thread not found. We will create thread.
        log.warn(f"Thread ID not found. Creating new Thread.\nError: {e}")
        thread = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
    # Create a thread message.
    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message.content
    )
    # Create a thread run.
    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    # Check if assistant requires action.
    run = await client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    if run.status == 'requires_action':
        # array of available tools for this assistant
        avail_tools = { "is_member_super_pal": is_member_super_pal }

        # retrieve tool function name, arguments, and call id
        tool_fn = avail_tools[run.required_action.
                            submit_tool_outputs.tool_calls[0].
                            function.name]
        tool_args = json.loads(run.required_action.
                            submit_tool_outputs.tool_calls[0].
                            function.arguments)
        tool_call_id = run.required_action.submit_tool_outputs.tool_calls[0].id

        # call tool function and save output
        tool_output = tool_fn(member=dict(tool_args).get('member'))

        # submit tools output
        run = await client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=[
                    {
                        "tool_call_id": tool_call_id,
                        "output": tool_output,
                    }
                ]
        )
        # Give 1 second for assistant to complete before first attempt.
        time.sleep(1)
        # check if assistant requires action again
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    # Retry every second.
    while run.status == 'in_progress' or run.status == 'queued':
        time.sleep(1)
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    if run.status == 'completed':
        # get most recent message from thread and post to discord channel
        messages = await client.beta.threads.messages.list(
            thread_id=thread.id
        )
        gpt_assistant_response = messages.data[0].content[0].text.value
        log.info(f"Super Pal Bot said \"{gpt_assistant_response}\"")
        return gpt_assistant_response
    else:
        log.info(f"Run status: {run.status}")

#############
# Bot setup #
#############
intents = discord.Intents.default()
intents.members = True         # Required to list all users in a guild.
intents.message_content = True # Required to use spin-the-wheel and grab winner.
bot = commands.Bot(command_prefix='!', intents=intents)

##################
# Slash commands #
##################
# Command: Promote users to "Super Pal of the Week"
@bot.tree.command(name='superpal')
@app_commands.describe(new_super_pal='the member you want to promote to super pal')
@app_commands.checks.has_role('Super Pal of the Week')
async def add_super_pal(interaction: discord.Interaction, new_super_pal: discord.Member) -> None:
    channel = bot.get_channel(CHANNEL_ID)
    role = discord.utils.get(interaction.guild.roles, name='Super Pal of the Week')
    # Promote new user and remove current super pal.
    # NOTE: I have to check for user role because commands.has_role() does not seem to work with app_commands
    if  role not in new_super_pal.roles:
        await new_super_pal.add_roles(role)
        await interaction.user.remove_roles(role)
        log.info(f'{new_super_pal.name} promoted by {interaction.user.name}.')
        await interaction.response.send_message(f'You have promoted {new_super_pal.mention} to super pal of the week!',
            ephemeral=True)
        await channel.send(f'Congratulations {new_super_pal.mention}! '
            f'You have been promoted to super pal of the week by {interaction.user.name}. {WELCOME_MSG}')
    else:
        await interaction.response.send_message(f'{new_super_pal.mention} is already super pal of the week.',
            ephemeral=True)      
 
###############
# Looped task #
###############
# Weekly Task: Choose "Super Pal of the Week"
@tasks.loop(hours=24*7)
async def super_pal_of_the_week():
    guild = bot.get_guild(GUILD_ID)
    channel = bot.get_channel(CHANNEL_ID)
    role = discord.utils.get(guild.roles, name='Super Pal of the Week')

    # Get list of members and filter out bots. Pick random member.
    true_member_list = [m for m in guild.members if not m.bot]
    spotw = random.choice(true_member_list)
    log.info(f'\nPicking new super pal of the week.')
    # Add super pal, remove current super pal, avoid duplicates.
    for member in true_member_list:
        if role in member.roles and member == spotw:
            log.info(f'{member.name} is already super pal of the week. Re-rolling.')
            return await super_pal_of_the_week()
        elif role in member.roles:
            log.info(f'{member.name} has been removed from super pal of the week role.')
            await member.remove_roles(role)
        elif member == spotw:
            log.info(f'{member.name} has been added to super pal of the week role.')
            await spotw.add_roles(role)
            await channel.send(f'Congratulations to {spotw.mention}, '
                f'the super pal of the week! {WELCOME_MSG}')

# Before Loop: Wait until Sunday at noon.
@super_pal_of_the_week.before_loop
async def before_super_pal_of_the_week():
    await bot.wait_until_ready()
    # Find amount of time until Sunday at noon.
    now = datetime.datetime.now()
    days_until_sunday = 7 - datetime.date.today().isoweekday()
    # If it's past noon on Sunday, add 7 days to timer.
    if datetime.date.today().isoweekday() == 7 and now.hour > 12:
        days_until_sunday = 7
    time_diff = now + datetime.timedelta(days = days_until_sunday)
    future = datetime.datetime(time_diff.year, time_diff.month, time_diff.day, 12, 0)
    # Sleep task until Sunday at noon.
    log.info(f'Sleeping for {(future-now)}. Will wake up Sunday at 12PM Eastern Time.')
    await asyncio.sleep((future-now).total_seconds())

##############
# Bot events #
##############
# Event: Avoid printing errors message for commands that aren't related to Super Pal Bot.
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CommandNotFound):
        return
    raise error

# Event: Start loop once bot is ready
@bot.event
async def on_ready():
    await bot.tree.sync()
    if not super_pal_of_the_week.is_running():
        super_pal_of_the_week.start()

# Event: Check Spin The Wheel rich message
@bot.event
async def on_message(message: discord.Message):
    guild = bot.get_guild(GUILD_ID)
    spin_the_wheel_role = discord.utils.get(guild.roles, name='Spin The Wheel')
    member = guild.get_member(message.author.id)
    # Reply to messages in Super Pal channel if they aren't commands and they aren't from a bot.
    if message.channel.id == CHANNEL_ID and message.content[0] != '!' and message.author.bot is False:
        gpt_response_msg = await respond_to_user(message)
        await message.channel.send(gpt_response_msg)
    # Only check embedded messages from Spin The Wheel Bot.
    if member is not None and spin_the_wheel_role in member.roles:
        embeds = message.embeds
        for embed in embeds:
            # Wait until message contains Spin the Wheel winner.
            if embed.description is None: continue
            elif embed.description[0] == '🏆':
                super_pal_role = discord.utils.get(guild.roles, name='Super Pal of the Week')
                # Grab winner name from Spin the Wheel message.
                new_super_pal_name = embed.description[12:-2]
                new_super_pal = discord.utils.get(guild.members, name=new_super_pal_name)
                log.info(f'{new_super_pal.name} was chosen by the wheel spin.')
                # Remove existing Super Pal of the Week
                true_member_list = [m for m in guild.members if not m.bot]
                for member in true_member_list:
                    if super_pal_role in member.roles:
                        await member.remove_roles(super_pal_role)
                # Add new winner to Super Pal of the Week.
                await new_super_pal.add_roles(super_pal_role)
                await message.channel.send(f'Congratulations {new_super_pal.mention}! '
                    f'You have been promoted to super pal of the week by wheel spin. {WELCOME_MSG}')
    # Handle commands if the message was not from Spin the Wheel.
    await bot.process_commands(message)

################
# Bot commands #
################
# Command: Spin the wheel for a random "Super Pal of the Week"
@bot.command(name='spinthewheel', pass_context=True)
@commands.has_role('Super Pal of the Week')
async def spinthewheel(ctx):
    guild = bot.get_guild(GUILD_ID)
    channel = bot.get_channel(CHANNEL_ID)

    role = discord.utils.get(guild.roles, name='Super Pal of the Week')
    # Get list of members and filter out bots.
    true_member_list = [m for m in guild.members if not m.bot]
    true_name_list = [member.name for member in true_member_list]
    true_name_str = ", ".join(true_name_list)
    # Send Spin the Wheel command.
    await channel.send(f'?pick {true_name_str}')
    log.info(f'\nSpinning the wheel for new super pal of the week.')

# Command: Promote users to "Super Pal of the Week"
@bot.command(name='spotw', pass_context=True)
@commands.has_role('Super Pal of the Week')
async def add_super_pal(ctx, new_super_pal: discord.Member):
    guild = bot.get_guild(GUILD_ID)
    channel = bot.get_channel(CHANNEL_ID)
    role = discord.utils.get(guild.roles, name='Super Pal of the Week')
    current_super_pal = ctx.message.author

    # Promote new user and remove current super pal.
    if role not in new_super_pal.roles:
        log.info(f'{new_super_pal.name} promoted by {current_super_pal.name}.')
        await new_super_pal.add_roles(role)
        await current_super_pal.remove_roles(role)
        await channel.send(f'Congratulations {new_super_pal.mention}! '
            f'You have been promoted to super pal of the week by {current_super_pal.name}. {WELCOME_MSG}')

# Command: Display more information about commands.
@bot.command(name='commands', pass_context=True)
@commands.has_role('Super Pal of the Week')
async def list_commands(ctx):
    log.info(f'{ctx.message.author.name} used help command.')
    channel = bot.get_channel(CHANNEL_ID)
    await channel.send(COMMANDS_MSG)

# Command: Send party parrot discord emoji.
@bot.command(name='cacaw', pass_context=True)
@commands.has_role('Super Pal of the Week')
async def cacaw(ctx):
    log.info(f'{ctx.message.author.name} used cacaw command.')
    channel = bot.get_channel(CHANNEL_ID)
    emoji_guild = bot.get_guild(EMOJI_GUILD_ID)
    partyparrot_emoji = discord.utils.get(emoji_guild.emojis, name='partyparrot')
    await channel.send(str(partyparrot_emoji)*50)

# Command: Randomly remove one user from voice chat
@bot.command(name='karatechop', pass_context=True)
@commands.has_role('Super Pal of the Week')
async def karate_chop(ctx):
    guild = bot.get_guild(GUILD_ID)
    channel = bot.get_channel(CHANNEL_ID)
    current_super_pal = ctx.message.author

    # Grab voice channels from env file values.
    voice_channels = [
        discord.utils.get(guild.voice_channels, name=voice_channel, type=discord.ChannelType.voice)
        for voice_channel in VOICE_CHANNELS
    ]
    active_members = [voice_channel.members for voice_channel in voice_channels]

    # Kick random user from voice channel.
    if not any(active_members):
        log.info(f'{current_super_pal.name} used karate chop, but no one is in the voice channels.')
        await channel.send(f'There is no one to karate chop, {current_super_pal.mention}!')
    else:
        log.info(f'{chopped_member.name} karate chopped')
        # Flatten user list, filter out bots, and choose random user
        flatten = lambda l: [x for y in l for x in y]
        true_member_list = [m for m in flatten(active_members) if not m.bot]
        chopped_member = random.choice(true_member_list)

        # Check that an 'AFK' channel exists and choose the first one we see
        afk_channels = [c.name for c in guild.voice_channels if 'AFK' in c.name]
        if any(afk_channels):
            await chopped_member.move_to(guild.voice_channels[afk_channels[0]])
            await channel.send(f'karate chopped {chopped_member.mention}!')
        else:
            await channel.send(f'{chopped_member.mention} would have been chopped, but an AFK channel was not found.\n'
                               f'Please complain to the server owner.')

# Command: Send party cat discord emoji
@bot.command(name='meow', pass_context=True)
@commands.has_role('Super Pal of the Week')
async def meow(ctx):
    log.info(f'{ctx.message.author.name} used meow command.')
    channel = bot.get_channel(CHANNEL_ID)
    emoji_guild = bot.get_guild(EMOJI_GUILD_ID)
    partymeow_emoji = discord.utils.get(emoji_guild.emojis, name='partymeow')
    await channel.send(str(partymeow_emoji)*50)

# Command: Surprise images (AI)
@bot.command(name='surprise', pass_context=True)
#@commands.has_role('Super Pal of the Week')
async def surprise(ctx):
    log.info(f'{ctx.message.author.name} used surprise command:\n\t{ctx.message.content}')
    channel = bot.get_channel(ART_CHANNEL_ID)
    your_text_here = ctx.message.content.removeprefix('!surprise ')
    # Talk to OpenAI image generation API.
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    try:
        response = await client.images.generate(
            prompt=your_text_here,
            n=4,
            response_format="b64_json",
            size="1024x1024"
        )
        if response['data']:
            await channel.send(files=[discord.File(io.BytesIO(base64.b64decode(img['b64_json'])),
                            filename='{random.randrange(1000)}.jpg') for img in response['data']])
        else:
            await channel.send('Failed to create surprise image. Everyone boo Adam.')
    except openai.APIError as err:
        log.warn(err)
        if str(err) == 'Your request was rejected as a result of our safety system.':
            await channel.send('Woah there nasty nelly, you asked for something too silly. OpenAI rejected your request due to "Safety". Please try again and be more polite next time.')
        elif str(err) == 'Billing hard limit has been reached':
            await channel.send('Adam is broke and can\'t afford this request.')

bot.run(TOKEN, log_handler=log_handler)
