import os
import discord
import openai
import json
import requests
import random
import asyncio
import dateparser
import datetime
from discord.ext import commands, tasks
from discord import app_commands


# =======================================================================================================================================
# =========================================================={GUILD/TESTING ID BLOCK}=====================================================
# =======================================================================================================================================

# insert the script in the text file here if the global script below is broken

# =======================================================================================================================================
# =========================================================={NO GUILD ID BLOCK}==========================================================
# =======================================================================================================================================

# --------------------------------------------------INITIALIZATION------------------------------------------------------

# Set your OpenAI API key (ensure this is set in your environment variables)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create a bot instance
intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix=';', intents=intents)

# Global dictionary to store welcome channel configurations
welcome_channels = {}

# Global dictionary to store temporary configuration data
temp_config = {}

# Status'
@tasks.loop(hours=1)  # Change status every hour
async def change_status():
    await bot.wait_until_ready()
    # Get the number of servers the bot is in
    num_servers = len(bot.guilds)
    # Define the statuses
    statuses = [
        discord.Activity(type=discord.ActivityType.watching, name="/commands"),
        discord.Game(f"in {num_servers} servers")
    ]
    # Choose a random status and set it
    current_status = random.choice(statuses) 
    await bot.change_presence(activity=current_status)

# Event listener for when the bot is ready
@bot.event
async def on_ready():
    # Sync the command tree globally
    await bot.tree.sync()
    global welcome_channels
    welcome_channels = await load_welcome_channels()
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print('------')
    change_status.start()


@bot.event
async def on_guild_join(guild):
    # Create an embed message
    embed = discord.Embed(
        title="Hello! I'm Kaydonbot",
        description="Thanks for inviting me to your server!",
        color=discord.Color.gold()
    )
    embed.add_field(name="Prefix", value="; for non-slash commands", inline=False)
    embed.add_field(name="Commands", value="Use `/commands` to see all my commands", inline=False)
    embed.set_footer(text="Kaydonbot - Copyright (c) Kayden Cormier -- K-GamesMedia")

       # List of preferred channel names
    preferred_channels = ["welcome", "general", "mod-chat", "mods-only"]

    # Try to find a preferred channel
    for channel_name in preferred_channels:
        channel = discord.utils.get(guild.text_channels, name=channel_name)
        if channel and channel.permissions_for(guild.me).send_messages:
            await channel.send(embed=embed)
            return

    # If no preferred channel is found, send in any channel where the bot has permission
    for channel in guild.text_channels:
        if channel.permissions_for(guild.me).send_messages:
            await channel.send(embed=embed)
            break


# -------------------------------------------------INITIALIZATION ENDS--------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------COMMANDS LIST-------------------------------------------------------


# Define a slash command for 'commands'
@bot.tree.command(name="commands", description="Get a list off all commands")
async def commands(interaction: discord.Interaction):
    await interaction.response.defer()
    message = await interaction.followup.send(embed=get_general_commands_embed())

    # Add reactions for navigation
    await message.add_reaction("⏪")  # Fast rewind to first page
    await message.add_reaction("⬅️")  # Previous page
    await message.add_reaction("➡️")  # Next page
    await message.add_reaction("⏩")  # Fast forward to last page

def get_general_commands_embed():
    embed = discord.Embed(
        title="Kaydonbot General Commands",
        description="Commands available for all users. Default prefix is ';'",
        color=discord.Color.gold()
    )
    embed.add_field(name="/commands", value="Displays list of all commands", inline=False)
    embed.add_field(name="/hello", value="Bot will say hello", inline=False)
    embed.add_field(name="/chat [prompt]", value="Sends a prompt to the GPT API and returns a response", inline=False)
    embed.add_field(name="/image [prompt]", value="Uses DALL-E 3 to generate an image based on your prompt", inline=False)
    embed.add_field(name="/quote", value="Get an inspirational quote", inline=False)
    embed.add_field(name="/joke", value="Tell a random joke", inline=False)
    embed.add_field(name="/weather [location]", value="Get the current weather for a location", inline=False)
    embed.add_field(name="/reminder [time] [reminder]", value="Set a reminder", inline=False)
    embed.add_field(name="/poll [question] [options]", value="Create a poll", inline=False)
    embed.add_field(name="/random [choices]", value="Make a random choice", inline=False)
    embed.set_footer(text="Page 1/4")
    return embed

def get_mod_commands_embed():
    embed = discord.Embed(
        title="Kaydonbot Moderator Commands",
        description="Commands available for moderators and administrators.",
        color=discord.Color.green()
    )
    # Add fields for each moderator command
    embed.add_field(name="/welcomeconfig", value="Configuration for user welcome message", inline=False)
    embed.add_field(name="/msgclear [channel] [number]", value="Clear a specified number of messages in a channel", inline=False)
    embed.add_field(name="/mute [member] [duration] [reason]", value="Mute a member", inline=False)
    embed.add_field(name="/unmute [member]", value="Unmute a member", inline=False)
    embed.add_field(name="/lock [channel]", value="Lock a channel", inline=False)
    embed.add_field(name="/unlock [channel]", value="Unlock a channel", inline=False)
    embed.add_field(name="/slowmode [channel] [seconds]", value="Set slowmode in a channel", inline=False)
    embed.add_field(name="/purgeuser [channel] [member] [number]", value="Clear messages by a specific user", inline=False)
    embed.add_field(name="/announce [channel] [message]", value="Send an announcement", inline=False)
    embed.add_field(name="/addrole [member] [role]", value="Add a role to a member", inline=False)
    embed.add_field(name="/removerole [member] [role]", value="Remove a role from a member", inline=False)
    embed.set_footer(text="Page 3/4")
    return embed

def get_bot_games_commands_embed():
    embed = discord.Embed(
        title="Kaydonbot Bot Games Commands",
        description="Fun games you can play with the bot.",
        color=discord.Color.blue()
    )
    embed.add_field(name="/battle", value="Start a battle game", inline=False)
    embed.add_field(name="/blackjack", value="Play a game of blackjack", inline=False)
    embed.add_field(name="/wouldyourather", value="Play a round of Would You Rather", inline=False)
    embed.add_field(name="/truthordare", value="Play a fun little Truth or Dare game", inline=False)
    # Add more bot game commands here
    embed.set_footer(text="Page 2/4")
    return embed

def get_suggestions_commands_embed():
    embed = discord.Embed(
        title="Kaydonbot Suggestions Commands",
        description="Commands to suggest new features or content for the bot.",
        color=discord.Color.purple()
    )
    embed.add_field(name="/cmdsuggestion [Suggestion]", value="Suggest a new command.", inline=False)
    embed.add_field(name="/tdsuggestion [option] {truth/dare} [suggestion]", value="Suggest a SFW Truth or Dare.", inline=False)
    embed.add_field(name="/wyrsuggestion [suggestion]", value="Suggest a 'Would You Rather' question.", inline=False)
    embed.set_footer(text="Page 4/4")
    return embed

@bot.event
async def on_reaction_add(reaction, user):
    if user != bot.user and reaction.message.author == bot.user:
        embeds = [
            get_general_commands_embed(), 
            get_bot_games_commands_embed(), 
            get_mod_commands_embed(),
            get_suggestions_commands_embed()
        ]
        current_page = int(reaction.message.embeds[0].footer.text.split('/')[0][-1]) - 1

        if reaction.emoji == "➡️":
            next_page = (current_page + 1) % len(embeds)
            await reaction.message.edit(embed=embeds[next_page])
        elif reaction.emoji == "⬅️":
            next_page = (current_page - 1) % len(embeds)
            await reaction.message.edit(embed=embeds[next_page])
        elif reaction.emoji == "⏩":
            await reaction.message.edit(embed=embeds[-1])  # Go to last page
        elif reaction.emoji == "⏪":
            await reaction.message.edit(embed=embeds[0])  # Go to first page

        await reaction.remove(user)

# --------------------------------------------------COMMANDS LIST ENDS-------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------SUGGESTIONS CMDS--------------------------------------------------------

# Ensure the suggestions directory exists
os.makedirs("~/hosting/suggestions", exist_ok=True)

@bot.tree.command(name="cmdsuggestion", description="Suggest a new command")
async def cmdsuggestion(interaction: discord.Interaction, suggestion: str):
    with open("~/hosting/suggestions/cmd_suggestions.txt", "a") as file:
        file.write(f"{suggestion}\n")
    await interaction.response.send_message("Your command suggestion has been recorded. Thank you!", ephemeral=True)

@bot.tree.command(name="tdsuggestion", description="Suggest a SFW Truth or Dare")
async def tdsuggestion(interaction: discord.Interaction, option: str, suggestion: str):
    filename = "truth_suggestions.txt" if option.lower() == "truth" else "dare_suggestions.txt"
    with open(f"~/hosting/suggestions/{filename}", "a") as file:
        file.write(f"{suggestion}\n")
    await interaction.response.send_message("Your Truth or Dare suggestion has been recorded. Thank you!", ephemeral=True)

@bot.tree.command(name="wyrsuggestion", description="Suggest a 'Would You Rather' question")
async def wyrsuggestion(interaction: discord.Interaction, suggestion: str):
    with open("~/hosting/suggestions/wyr_suggestions.txt", "a") as file:
        file.write(f"{suggestion}\n")
    await interaction.response.send_message("Your 'Would You Rather' suggestion has been recorded. Thank you!", ephemeral=True)




# ----------------------------------------------------SUGGESTIONS ENDS-------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------MOD-ONLY COMMANDS-------------------------------------------------------

# Check if user is admin/mod
def is_admin_or_mod():
    async def predicate(interaction: discord.Interaction):
        return interaction.user.guild_permissions.administrator or \
               any(role.name.lower() in ['admin', 'moderator'] for role in interaction.user.roles)
    return app_commands.check(predicate)

#******************************WELCOME MESSAGE******************************

def save_welcome_channels():
    try:
        with open('welcome_channels.json', 'w') as file:
            json.dump(welcome_channels, file, indent=4)
    except Exception as e:
        print(f"Error saving welcome channels: {e}")
        # Consider logging this error or handling it appropriately

async def load_welcome_channels():
    global welcome_channels
    try:
        with open('welcome_channels.json', 'r') as file:
            welcome_channels = json.load(file)
    except FileNotFoundError:
        welcome_channels = {}
        # Consider logging this error or handling it appropriately

@bot.tree.command(name="welcomeconfig", description="Configure the welcome channel")
@is_admin_or_mod()
async def welcomeconfig(interaction: discord.Interaction):
    try:
        await interaction.response.defer()

        # Initiate the configuration process
        temp_config[interaction.guild_id] = {"stage": 1}  # Stage 1: Ask to enable/disable

        embed = discord.Embed(
            title="Welcome Configuration",
            description="Welcome to the welcome config settings.\n\n"
                        "1. Please type 'enable' to enable welcome messages or 'disable' to disable them.\n"
                        "2. If enabled, you will be prompted to specify a channel and set a custom welcome message.",
            color=discord.Color.gold()
        )
        await interaction.followup.send(embed=embed)
    except Exception as e:
        await interaction.followup.send(f"Failed to initiate welcome configuration: {e}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    guild_id = message.guild.id
    if guild_id in temp_config:
        if temp_config[guild_id]["stage"] == 1:
            # Handle enabling/disabling welcome messages
            content_lower = message.content.strip().lower()
            if content_lower == 'enable':
                temp_config[guild_id] = {"stage": 2, "enabled": True}  # Move to stage 2
                await message.channel.send("Welcome messages enabled. Please mention the channel for welcome messages.")
            elif content_lower == 'disable':
                welcome_channels[guild_id] = {"enabled": False}
                save_welcome_channels()
                await message.channel.send("Welcome messages will be disabled. They can always be enabled later.")
                del temp_config[guild_id]
            else:
                await message.channel.send("Please type 'enable' or 'disable'.")

        elif temp_config[guild_id]["stage"] == 2:
            # Handle channel selection
            if message.channel_mentions:
                selected_channel = message.channel_mentions[0]
                temp_config[guild_id] = {"stage": 3, "channel_id": selected_channel.id, "enabled": True}  # Move to stage 3
                embed = discord.Embed(
                    title="Welcome Configuration",
                    description="Channel set successfully. Please specify the custom welcome message.",
                    color=discord.Color.green()
                )
                await message.channel.send(embed=embed) 
            else:
                await message.channel.send("Please mention a valid channel.")

        elif temp_config[guild_id]["stage"] == 3:
            # Handle custom welcome message
            custom_message = message.content
            channel_id = temp_config[guild_id]["channel_id"]
            welcome_channels[guild_id] = {"channel_id": channel_id, "message": custom_message, "enabled": True}
            save_welcome_channels()  # Save the configuration

            embed = discord.Embed(
                title="Welcome Configuration",
                description="Custom welcome message set successfully.",
                color=discord.Color.gold()
            )
            await message.channel.send(embed=embed)

            # Clear temporary configuration data
            del temp_config[guild_id]

# Send welcome message on user join
@bot.event
async def on_member_join(member):
    guild_id = member.guild.id
    if guild_id in welcome_channels and welcome_channels[guild_id].get("enabled", False):
        channel_id = welcome_channels[guild_id].get("channel_id")
        custom_message = welcome_channels[guild_id].get("message", f"Welcome to the server, {member.mention}!")
        channel = member.guild.get_channel(channel_id) if channel_id else None

        if channel:
            await channel.send(custom_message.format(member=member.mention))
    else:
        # Fallback to default message if no custom configuration is found or welcome messages are disabled
        channel = discord.utils.get(member.guild.text_channels, name='welcome')
        if channel:
            await channel.send(f"Welcome to the server, {member.mention}!")

#****************************WELCOME MESSAGE ENDS****************************

# Define a slash command for 'msgclear'
@bot.tree.command(name="msgclear", description="Clear a specified number of messages in a channel")
@is_admin_or_mod()
async def msgclear(interaction: discord.Interaction, channel: discord.TextChannel, number: int):
    try:
        await interaction.response.defer()

        if number < 1 or number > 100:
            await interaction.followup.send("Please specify a number between 1 and 100.")
            return

        messages = [message async for message in channel.history(limit=number)]
        if not messages:
            await interaction.followup.send("No messages to delete.")
            return

        deleted_count = 0
        for message in messages:
            if (discord.utils.utcnow() - message.created_at).days < 14:
                await message.delete()
                deleted_count += 1

        confirmation_message = await interaction.followup.send(f"Cleared {deleted_count} messages in {channel.mention}.")
        await asyncio.sleep(5)  # Wait for 5 seconds
        await confirmation_message.delete()
    except Exception as e:
        await interaction.followup.send(f"Failed to clear messages: {e}")


@bot.tree.command(name="warn", description="Warn a member")
@is_admin_or_mod()
async def warn(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided"):
    try:
        await interaction.response.defer()
        # Send a DM to the member with the warning
        await member.send(f"You have been warned for: {reason}")
        await interaction.followup.send(f"{member.mention} has been warned for: {reason}")
    except Exception as e:
        await interaction.followup.send(f"Failed to warn member: {e}")

@bot.tree.command(name="kick", description="Kick a member from the server")
@is_admin_or_mod()
async def kick(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided"):
    try:
        await interaction.response.defer()
        await member.kick(reason=reason)
        await interaction.followup.send(f"{member.mention} has been kicked for: {reason}")
    except Exception as e:
        await interaction.followup.send(f"Failed to kick member: {e}")

@bot.tree.command(name="ban", description="Ban a member from the server")
@is_admin_or_mod()
async def ban(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided"):
    try:
        await interaction.response.defer()
        await member.ban(reason=reason)
        await interaction.followup.send(f"{member.mention} has been banned for: {reason}")
    except Exception as e:
        await interaction.followup.send(f"Failed to ban member: {e}")

import asyncio

@bot.tree.command(name="mute", description="Mute a member")
@is_admin_or_mod()
async def mute(interaction: discord.Interaction, member: discord.Member, duration: int, reason: str = "No reason provided"):
    try:
        await interaction.response.defer()
        muted_role = discord.utils.get(member.guild.roles, name="Muted")
        if not muted_role:
            await interaction.followup.send("Muted role not found.")
            return

        await member.add_roles(muted_role, reason=reason)
        await interaction.followup.send(f"{member.mention} has been muted for {duration} minutes. Reason: {reason}")

        await asyncio.sleep(duration * 60)  # Convert minutes to seconds
        if muted_role in member.roles:
            await member.remove_roles(muted_role, reason="Mute duration expired")
            await interaction.followup.send(f"{member.mention} has been unmuted.")
    except Exception as e:
        await interaction.followup.send(f"Failed to mute member: {e}")

@bot.tree.command(name="unmute", description="Unmute a member")
@is_admin_or_mod()
async def unmute(interaction: discord.Interaction, member: discord.Member):
    try:
        await interaction.response.defer()
        muted_role = discord.utils.get(member.guild.roles, name="Muted")
        if not muted_role:
            await interaction.followup.send("Muted role not found.")
            return

        if muted_role in member.roles:
            await member.remove_roles(muted_role, reason="Manually unmuted")
            await interaction.followup.send(f"{member.mention} has been unmuted.")
        else:
            await interaction.followup.send(f"{member.mention} is not muted.")
    except Exception as e:
        await interaction.followup.send(f"Failed to unmute member: {e}")


@bot.tree.command(name="lock", description="Lock a channel")
@is_admin_or_mod()
async def lock(interaction: discord.Interaction, channel: discord.TextChannel):
    try:
        await interaction.response.defer()
        await channel.set_permissions(channel.guild.default_role, send_messages=False)
        await interaction.followup.send(f"{channel.mention} has been locked.")
    except Exception as e:
        await interaction.followup.send(f"Failed to lock the channel: {e}")

@bot.tree.command(name="unlock", description="Unlock a channel")
@is_admin_or_mod()
async def unlock(interaction: discord.Interaction, channel: discord.TextChannel):
    try:
        await interaction.response.defer()
        await channel.set_permissions(channel.guild.default_role, send_messages=True)
        await interaction.followup.send(f"{channel.mention} has been unlocked.")
    except Exception as e:
        await interaction.followup.send(f"Failed to unlock the channel: {e}")


@bot.tree.command(name="slowmode", description="Set slowmode in a channel")
@is_admin_or_mod()
async def slowmode(interaction: discord.Interaction, channel: discord.TextChannel, seconds: int):
    try:
        await interaction.response.defer()
        await channel.edit(slowmode_delay=seconds)
        await interaction.followup.send(f"Slowmode set to {seconds} seconds in {channel.mention}.")
    except Exception as e:
        await interaction.followup.send(f"Failed to set slowmode: {e}")


@bot.tree.command(name="purgeuser", description="Clear messages by a specific user")
@is_admin_or_mod()
async def purgeuser(interaction: discord.Interaction, channel: discord.TextChannel, member: discord.Member, number: int):
    try:
        await interaction.response.defer()
        deleted_count = 0
        async for message in channel.history(limit=200):
            if message.author == member and deleted_count < number:
                await message.delete()
                deleted_count += 1
            if deleted_count >= number:
                break
        await interaction.followup.send(f"Cleared {deleted_count} messages from {member.mention} in {channel.mention}.")
    except Exception as e:
        await interaction.followup.send(f"Failed to clear messages: {e}")


@bot.tree.command(name="announce", description="Send an announcement")
@is_admin_or_mod()
async def announce(interaction: discord.Interaction, channel: discord.TextChannel, message: str):
    try:
        await interaction.response.defer()
        await channel.send(message)
        await interaction.followup.send(f"Announcement sent in {channel.mention}.")
    except Exception as e:
        await interaction.followup.send(f"Failed to send announcement: {e}")


@bot.tree.command(name="addrole", description="Add a role to a member")
@is_admin_or_mod()
async def addrole(interaction: discord.Interaction, member: discord.Member, role: discord.Role):
    try:
        await interaction.response.defer()
        if role in member.roles:
            await interaction.followup.send(f"{member.mention} already has the {role.name} role.")
            return

        await member.add_roles(role)
        await interaction.followup.send(f"Added {role.name} role to {member.mention}.")
    except Exception as e:
        await interaction.followup.send(f"Failed to add role: {e}")


@bot.tree.command(name="removerole", description="Remove a role from a member")
@is_admin_or_mod()
async def removerole(interaction: discord.Interaction, member: discord.Member, role: discord.Role):
    try:
        await interaction.response.defer()
        if role not in member.roles:
            await interaction.followup.send(f"{member.mention} does not have the {role.name} role.")
            return

        await member.remove_roles(role)
        await interaction.followup.send(f"Removed {role.name} role from {member.mention}.")
    except Exception as e:
        await interaction.followup.send(f"Failed to remove role: {e}")



# -------------------------------------------------MOD-ONLY COMMANDS ENDS----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------OPENAI COMMANDS---------------------------------------------------------


# Define a slash command for 'chat'
@bot.tree.command(name="chat", description="Get a response from GPT")
async def chat(interaction: discord.Interaction, prompt: str):
    # Prepare the chat messages for the API call
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # List of models to try in order
    models = ["gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo"]
    response_sent = False

    for model in models:
        try:
            # Call OpenAI Chat Completions API with the current model
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )

            # Send the response back to Discord and mark as sent
            await interaction.response.send_message(response['choices'][0]['message']['content'])
            response_sent = True
            break
        except Exception as e:
            # If there's an error (like model not available), continue to the next model
            continue

    # If no response was sent, notify the user
    if not response_sent:
        await interaction.response.send_message("Sorry, I'm unable to get a response at the moment.")

# Function to call DALL-E 3 API
async def generate_dalle_image(prompt: str):
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Define a slash command for 'image'
@bot.tree.command(name="image", description="Generate an image using DALL-E 3")
async def image(interaction: discord.Interaction, prompt: str):
    # Defer the response to give more time for processing
    await interaction.response.defer()

    image_url = await generate_dalle_image(prompt)
    if image_url:
        await interaction.followup.send(image_url)
    else:
        await interaction.followup.send("Sorry, I couldn't generate an image.")

# ------------------------------------------------OPENAI COMMANDS ENDS-----------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------GENERAL COMMANDS--------------------------------------------------------

# Define a slash command for 'hello'
@bot.tree.command(name="hello", description="This is just a simple hello command.")  
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message("Hello! How are you today?")

@bot.tree.command(name="userinfo", description="Get information about a user")
async def userinfo(interaction: discord.Interaction, member: discord.Member):
    try:
        await interaction.response.defer()
        embed = discord.Embed(title=f"User Info for {member}", color=discord.Color.blue())
        embed.add_field(name="Username", value=str(member), inline=True)
        embed.add_field(name="ID", value=member.id, inline=True)
        embed.add_field(name="Joined at", value=member.joined_at.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
        embed.add_field(name="Roles", value=" ".join([role.mention for role in member.roles[1:]]), inline=False)
        embed.add_field(name="Status", value=str(member.status).title(), inline=True)
        await interaction.followup.send(embed=embed)
    except Exception as e:
        await interaction.followup.send(f"Failed to retrieve user info: {e}")

    
@bot.tree.command(name="serverinfo", description="Get information about the server")
async def serverinfo(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        guild = interaction.guild
        embed = discord.Embed(title=f"Server Info for {guild.name}", color=discord.Color.green())
        embed.set_thumbnail(url=guild.icon_url)
        embed.add_field(name="Server ID", value=guild.id, inline=True)
        embed.add_field(name="Owner", value=guild.owner.mention, inline=True)
        embed.add_field(name="Members", value=guild.member_count, inline=True)
        embed.add_field(name="Created at", value=guild.created_at.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
        embed.add_field(name="Roles", value=", ".join([role.name for role in guild.roles[1:]]), inline=False)
        await interaction.followup.send(embed=embed)
    except Exception as e:
        await interaction.followup.send(f"Failed to retrieve server info: {e}")

    
@bot.tree.command(name="poll", description="Create a poll")
async def poll(interaction: discord.Interaction, question: str, options_str: str):
    try:
        await interaction.response.defer()
        options = options_str.split(",")  # Split the options string by commas
        if len(options) < 2:
            await interaction.followup.send("Please provide at least two options for the poll, separated by commas.")
            return

        embed = discord.Embed(title="Poll", description=question, color=discord.Color.blue())
        reactions = ['🔵', '🔴', '🟢', '🟡', '🟣', '🟠', '⚫', '⚪']  # Add more if needed

        poll_options = {reactions[i]: option.strip() for i, option in enumerate(options) if i < len(reactions)}
        for emoji, option in poll_options.items():
            embed.add_field(name=emoji, value=option, inline=False)

        poll_message = await interaction.followup.send(embed=embed)
        for emoji in poll_options.keys():
            await poll_message.add_reaction(emoji)
    except Exception as e:
        await interaction.followup.send(f"Failed to create poll: {e}")


@bot.tree.command(name="random", description="Make a random choice")
async def random_choice(interaction: discord.Interaction, choices_str: str):
    try:
        await interaction.response.defer()
        choices = choices_str.split(",")  # Split the choices string by commas
        if len(choices) < 2:
            await interaction.followup.send("Please provide at least two choices, separated by commas.")
            return

        selected_choice = random.choice(choices).strip()  
        await interaction.followup.send(f"Randomly selected: {selected_choice}")
    except Exception as e:
        await interaction.followup.send(f"Failed to make a random choice: {e}")


@bot.tree.command(name="weather", description="Get the current weather for a location")
async def weather(interaction: discord.Interaction, location: str):
    try:
        await interaction.response.defer()
        api_key = os.getenv("OPENWEATHER_API_KEY")  # Fetch the API key from an environment variable
        if not api_key:
            await interaction.followup.send("Weather API key not set.")
            return

        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url).json()

        if response.get("cod") != 200:
            await interaction.followup.send(f"Failed to retrieve weather info for {location}.")
            return

        weather_description = response['weather'][0]['description']
        temperature = response['main']['temp']
        humidity = response['main']['humidity']
        wind_speed = response['wind']['speed']

        weather_info = (f"Weather in {location.title()}: {weather_description}\n"
                        f"Temperature: {temperature}°C\n"
                        f"Humidity: {humidity}%\n"
                        f"Wind Speed: {wind_speed} m/s")

        await interaction.followup.send(weather_info)
    except Exception as e:
        await interaction.followup.send(f"Failed to retrieve weather info: {e}")
    
@bot.tree.command(name="reminder", description="Set a reminder")
async def reminder(interaction: discord.Interaction, time: str, reminder: str):
    try:
        await interaction.response.defer()
        # Parse the time string into a datetime object
        reminder_time = dateparser.parse(time)
        if not reminder_time:
            await interaction.followup.send("Invalid time format.")
            return

        # Calculate the delay in seconds
        delay = (reminder_time - datetime.datetime.now()).total_seconds()
        if delay < 0:
            await interaction.followup.send("Time is in the past.")
            return

        # Wait for the specified time and send the reminder
        await asyncio.sleep(delay)
        await interaction.followup.send(f"Reminder: {reminder}")
    except Exception as e:
        await interaction.followup.send(f"Failed to set reminder: {e}")


@bot.tree.command(name="quote", description="Get an inspirational quote")
async def quote(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        response = requests.get("https://zenquotes.io/api/random")
        if response.status_code != 200:
            await interaction.followup.send("Failed to retrieve a quote.")
            return

        quote_data = response.json()[0]
        quote_text = f"{quote_data['q']} - {quote_data['a']}"
        await interaction.followup.send(quote_text)
    except Exception as e:
        await interaction.followup.send(f"Failed to retrieve a quote: {e}")
    
@bot.tree.command(name="joke", description="Tell a random joke")
async def joke(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        headers = {'Accept': 'application/json'}
        response = requests.get("https://icanhazdadjoke.com/", headers=headers)
        if response.status_code != 200:
            await interaction.followup.send("Failed to retrieve a joke.")
            return

        joke_text = response.json()['joke']
        await interaction.followup.send(joke_text)
    except Exception as e:
        await interaction.followup.send(f"Failed to retrieve a joke: {e}")

# ------------------------------------------------GENERAL COMMANDS ENDS----------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------BOT GAMES------------------------------------------------------------


# _________________________________________________BLACKJACK_____________________________________________

# Define card values
card_values = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

# Function to draw a card
def draw_card():
    card = random.choice(list(card_values.keys()))
    suit = random.choice(['♠', '♦', '♣', '♥'])
    return f"{card}{suit}"

# Function to calculate the score of a hand
def calculate_score(hand):
    score = sum(card_values[card[:-1]] for card in hand)
    # Adjust for Aces
    for card in hand:
        if card[:-1] == 'A' and score > 21:
            score -= 10
    return score

# Function to check for Blackjack
def is_blackjack(hand):
    return calculate_score(hand) == 21 and len(hand) == 2

# Function to update the game message
async def update_game_message(message, player_hand, dealer_hand, game_over=False):
    player_score = calculate_score(player_hand)
    dealer_score = calculate_score(dealer_hand) if game_over else '?'
    dealer_display = " ".join(dealer_hand) if game_over else dealer_hand[0] + " ?"

    embed = discord.Embed(title="Blackjack", color=discord.Color.green())
    embed.add_field(name="Your Hand", value=" ".join(player_hand) + f" (Score: {player_score})", inline=False)
    embed.add_field(name="Dealer's Hand", value=dealer_display + f" (Score: {dealer_score})", inline=False)

    if game_over:
        if player_score > 21:
            embed.set_footer(text="You busted! Dealer wins.")
        elif dealer_score > 21 or player_score > dealer_score:
            embed.set_footer(text="You win!")
        elif player_score == dealer_score:
            embed.set_footer(text="It's a tie!")
        else:
            embed.set_footer(text="Dealer wins.")

    await message.edit(embed=embed)

# Blackjack command
@bot.tree.command(name="blackjack", description="Play a game of blackjack")
async def blackjack(interaction: discord.Interaction):
    player_hand = [draw_card(), draw_card()]
    dealer_hand = [draw_card(), draw_card()]

    # Check for Blackjack on initial deal
    if is_blackjack(player_hand) or is_blackjack(dealer_hand):
        await interaction.response.send_message("Checking for Blackjack...")
        await update_game_message(message, player_hand, dealer_hand, game_over=True)
        return

    message = await interaction.response.send_message("Starting Blackjack game...")
    await update_game_message(message, player_hand, dealer_hand)

    # Add reactions for player actions
    await message.add_reaction('♠')  # Hit
    await message.add_reaction('♦')  # Stand

    def check(reaction, user):
        return user == interaction.user and str(reaction.emoji) in ['♠', '♦'] and reaction.message.id == message.id

    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=60.0, check=check)

        if str(reaction.emoji) == '♠':  # Hit
            player_hand.append(draw_card())
            if calculate_score(player_hand) > 21:
                await update_game_message(message, player_hand, dealer_hand, game_over=True)
            else:
                await update_game_message(message, player_hand, dealer_hand)
        elif str(reaction.emoji) == '♦':  # Stand
            while calculate_score(dealer_hand) < 17:
                dealer_hand.append(draw_card())
            await update_game_message(message, player_hand, dealer_hand, game_over=True)

    except asyncio.TimeoutError:
        await message.clear_reactions()
        await message.edit(content="Blackjack game timed out.", embed=None)
# _________________________________________________BLACKJACK ENDS_____________________________________________

# _________________________________________________BATTLE GAME________________________________________________

# Global dictionary to store game states
game_states = {}

# Define the battle command
@bot.tree.command(name="battle", description="Start a battle game")
async def battle(interaction: discord.Interaction):
    player_health = 100
    bot_health = 100
    embed = discord.Embed(title="Battle Game", description="Choose your action!", color=discord.Color.red())
    embed.add_field(name="Your Health", value=str(player_health), inline=True)
    embed.add_field(name="Bot's Health", value=str(bot_health), inline=True)
    embed.add_field(name="Actions", value="⚔️ to attack\n🛡️ to defend", inline=False)

    message = await interaction.response.send_message(embed=embed)

    # Add reactions for game actions
    await message.add_reaction('⚔️')  # Attack
    await message.add_reaction('🛡️')  # Defend

    # Store initial game state
    game_states[message.id] = {
        "player_health": player_health,
        "bot_health": bot_health,
        "interaction": interaction
    }

# Handle reactions
@bot.event
async def on_reaction_add_battle(reaction, user):
    if user != bot.user and reaction.message.id in game_states:
        game_state = game_states[reaction.message.id]
        interaction = game_state["interaction"]

        if user.id != interaction.user.id:
            return  # Ignore reactions from other users

        player_action = reaction.emoji
        bot_action = random.choice(['⚔️', '🛡️'])

        # Determine the outcome of the turn
        if player_action == '⚔️' and bot_action == '⚔️':
            game_state["player_health"] -= 10
            game_state["bot_health"] -= 10
        elif player_action == '⚔️' and bot_action == '🛡️':
            game_state["bot_health"] -= 5
        elif player_action == '🛡️' and bot_action == '⚔️':
            game_state["player_health"] -= 5

        # Update the embed with the new health values
        embed = discord.Embed(title="Battle Game", description="Choose your action!", color=discord.Color.red())
        embed.add_field(name="Your Health", value=str(game_state["player_health"]), inline=True)
        embed.add_field(name="Bot's Health", value=str(game_state["bot_health"]), inline=True)
        embed.add_field(name="Bot's Action", value="Bot chose to " + ("attack" if bot_action == '⚔️' else "defend"), inline=False)

        await reaction.message.edit(embed=embed)

        # Check for end of game
        if game_state["player_health"] <= 0 or game_state["bot_health"] <= 0:
            winner = "You win!" if game_state["player_health"] > game_state["bot_health"] else "Bot wins!"
            await reaction.message.edit(content=winner, embed=None)
            del game_states[reaction.message.id]  # Clean up the game state
            return

        # Prepare for the next turn
        await reaction.message.clear_reactions()
        await reaction.message.add_reaction('⚔️')  # Attack
        await reaction.message.add_reaction('🛡️')  # Defend
# _________________________________________________BATTLE GAME ENDS________________________________________________


# _________________________________________________WOULD YOU RATHER________________________________________________


# Load Would You Rather questions from JSON file
def load_wyr_questions():
    with open('wouldyourather.json', 'r') as file:
        return json.load(file)

# Define the Would You Rather command
@bot.tree.command(name="wouldyourather", description="Play 'Would You Rather'")
async def wouldyourather(interaction: discord.Interaction):
    await interaction.response.defer()
    questions = load_wyr_questions()
    question = random.choice(questions)

    embed = discord.Embed(title="Would You Rather", description=question["question"], color=discord.Color.blue())
    embed.add_field(name="Option 1", value=question["option1"], inline=False)
    embed.add_field(name="Option 2", value=question["option2"], inline=False)

    message = await interaction.followup.send(embed=embed)  # Use followup.send

    # Add reactions for options
    await message.add_reaction("1️⃣")  # Option 1
    await message.add_reaction("2️⃣")  # Option 2

    # Wait for a reaction
    def wyr_check(reaction, user):
        return user == interaction.user and str(reaction.emoji) in ["1️⃣", "2️⃣"] and reaction.message.id == message.id

    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=60.0, check=wyr_check)
        choice_key = "option1" if str(reaction.emoji) == "1️⃣" else "option2"
        await interaction.followup.send(f"{user.mention} chose {choice_key.replace('option', 'Option ')}: {question[choice_key]}")
    except asyncio.TimeoutError:
        await message.clear_reactions()
        await message.edit(content="Would You Rather game timed out.", embed=None)

# _________________________________________________WOULD YOU RATHER ENDS____________________________________________

# ______________________________________________________TRUTH OR DARE_______________________________________________

# Load Truth or Dare questions from JSON file
def load_tod_questions():
    with open('truthordare.json', 'r') as file:
        return json.load(file)

# Define the Truth or Dare command
@bot.tree.command(name="truthordare", description="Play 'Truth or Dare'")
async def truth_or_dare(interaction: discord.Interaction):
    await interaction.response.defer()
    questions = load_tod_questions()

    embed = discord.Embed(title="Truth or Dare", description="React with 🤔 for Truth or 😈 for Dare", color=discord.Color.blue())
    message = await interaction.followup.send(embed=embed)

    if message:
        await message.add_reaction("🤔")  # Truth
        await message.add_reaction("😈")  # Dare

    # Wait for a reaction
    def tod_check(reaction, user):
        return user == interaction.user and str(reaction.emoji) in ["🤔", "😈"] and reaction.message.id == message.id

    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=60.0, check=tod_check)
        if str(reaction.emoji) == "🤔":
            selected = random.choice(questions['truths'])
            response_type = "Truth"
        else:
            selected = random.choice(questions['dares'])
            response_type = "Dare"

        response_embed = discord.Embed(
            title=f"{response_type} for {user.display_name}",
            description=selected,
            color=discord.Color.green() if response_type == "Truth" else discord.Color.red()
        )
        await interaction.followup.send(embed=response_embed)
    except asyncio.TimeoutError:
        await message.clear_reactions()
        await message.edit(content="Truth or Dare game timed out.", embed=None)

# ________________________________________________TRUTH OR DARE ENDS________________________________________________

# --------------------------------------------------BOT GAMES END----------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------BOT TOKEN BELOW---------------------------------------------------------



# Run the bot with your token
bot.run(os.getenv('DISCORD_BOT_TOKEN'))

# --------------------------------------------------BOT TOKEN ENDS--------------------------------------------------------

# ========================================================================================================================
