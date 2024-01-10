import re

import discord
from discord.app_commands import tree
from discord.ext import commands

from src.command_enums import Command
from src.langchain_wrapper import LangChainGpt


def message_to_dict(message):
    return {
        'guild_id': message.guild.id,
        'channel_id': message.channel.id,
        'name': message.author.name,
        'author_id': message.author.id,
        'id': message.id,
        'content': message.content,
    }


class DiscordBot(discord.ext.commands.Bot):
    def __init__(self, chat_gpt: LangChainGpt):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(command_prefix='$', intents=intents)
        self.chat_gpt = chat_gpt
        self.log_channel = None
        self.inactive_members = []
        self.active_members = []
        self.channel_messages = []
        self.members = []
        self.message_counts = {}
        self.infant_channel = 1148798512750923797
        self.kindergarten_channel = 1150088312732790784

        @self.command()
        @commands.has_role('Parent')
        async def show(ctx, arg):
            ghosts_string_list = []
            for member in self.members:
                member_roles = [role.name for role in member.roles]
                if arg in member_roles:
                    ghosts_string_list.append(f'- {member.name}')
            ghosts_string = '\n'.join(ghosts_string_list)
            if len(ghosts_string) == 0:
                ghosts_string = 'Not found'
            print(ghosts_string)
            await self.log_channel.send(ghosts_string)

        @self.command()
        @commands.has_role('Parent')
        async def assign(ctx, arg):
            await self.load_msgs(ctx)
            if arg == 'Toddler':
                print('Assigning Toddler role to active members...')
                await self.log_channel.send('Assigning Toddler role to active members...')
                await self.assign_roles_to_active_members(ctx.guild, arg)
            if arg == 'Infant':
                print('Assigning Baby role to inactive members...')
                await self.log_channel.send('Assigning Infant role to inactive members...')
                await self.assign_roles_to_inactive_members(ctx.guild, arg)
            await self.log_channel.send('Done')

        @self.command()
        @commands.has_role('Parent')
        async def refresh(ctx):
            print('Retrieving all channel messages...')
            await self.log_channel.send('Retrieving all channel messages...')
            self.channel_messages = await self.retrieve_channel_msg(ctx.guild)
            print('Done')
            await self.log_channel.send('Done')

        @self.command()
        @commands.has_role('Parent')
        async def count(ctx, arg):
            await self.load_msgs(ctx)

            message_counts = self.count_user_messages(self.channel_messages, self.members)
            if arg == 'inactive':
                count_condition = lambda count: count == 0
                count_message = 'inactive'
                members_list = self.inactive_members
            elif arg == 'active':
                count_condition = lambda count: count > 0
                count_message = 'active'
                members_list = self.active_members
            else:
                count_condition = lambda count: True
                count_message = 'all'
                members_list = self.members

            members_list.clear()
            members_list.extend([member for member in self.members if count_condition(message_counts[member.name])])

            print(f'Number of {count_message} members: {len(members_list)}')
            await self.log_channel.send(f'Number of {count_message} members: {len(members_list)}')

    async def fetch_all_members(self, guild):
        async for member in guild.fetch_members(limit=None):
            if not member.bot and [role for role in member.roles if role.name != 'Admin']:
                self.members.append(member)
        return self.members

    async def load_msgs(self, ctx):
        if len(self.channel_messages) == 0:
            print('No channel messages saved in memory. Retrieving all channel messages...')
            await self.log_channel.send('No channel messages saved in memory. Retrieving all channel messages...')
            async with ctx.typing():
                self.channel_messages = await self.retrieve_channel_msg(ctx.guild)
                print('Done')
                await self.log_channel.send('Done')

    async def on_ready(self):
        print('Logged on as', self.user)
        await self.tree.sync(guild=discord.Object(id=1030501230797131887))
        for guild in self.guilds:
            print(f'Connected to {guild.name}')
            await self.tree.sync()
            self.members = await self.fetch_all_members(guild)
            self.log_channel = guild.get_channel(1148894899358404618)
            # print('Retrieving all channel messages...')
            # await self.log_channel.send('Retrieving all channel messages...')
            # self.channel_messages = await self.retrieve_channel_msg(guild)
            # print('Done')
            # await self.log_channel.send('Ready for commands!')

    def count_user_messages(self, channel_messages, members):
        for member in members:
            self.message_counts[member.name] = 0
        for message in channel_messages:
            if message.author.name in self.message_counts:
                self.message_counts[message.author.name] += 1
        return self.message_counts

    async def assign_roles_to_active_members(self, guild, role_name='Toddler'):
        for member in self.members:
            if member not in self.inactive_members and role_name not in [role.name for role in member.roles]:
                await self.assign_role(guild, member, role_name)

    async def assign_roles_to_inactive_members(self, guild, role_name='Infant'):
        for member in self.inactive_members:
            if role_name not in [role.name for role in member.roles]:
                await self.assign_role(guild, member, role_name)

    async def assign_role(self, guild, member, role_name):
        role = discord.utils.get(guild.roles, name=role_name)
        await member.add_roles(role)

    async def remove_role(self, guild, member, role_name):
        role = discord.utils.get(guild.roles, name=role_name)
        await member.remove_roles(role)

    async def retrieve_channel_msg(self, guild):
        for channel in guild.text_channels:
            async for message in channel.history(limit=None):
                if message.content != "" and not message.author.bot:
                    self.channel_messages.append(message)
        return self.channel_messages

    async def on_interaction(self, interaction: discord.Interaction):
        if interaction.type == discord.InteractionType.application_command:
            given_command = interaction.data['name']
            if given_command == Command.CHAT.value:
                await interaction.response.defer()
                message = interaction.data['options'][0]['value']
                user = interaction.user
                print(f'\n\n{user.name}: {message}')
                print("LangChain: ")
                answer = self.chat_gpt.predict(f'"ユーザー名："{user} {message}')
                response = f'{user.name}: {message} \n\n MSBN: {answer}'
                await interaction.followup.send(response)

    async def on_message(self, message):
        author, content = message.author, message.content
        sanitized_content = re.sub("<@\d+>", "", content).strip()

        if not self.is_message_empty(sanitized_content) and not author.bot and \
                message.channel.id in [self.infant_channel, self.kindergarten_channel]:
            print(f'\n{author.name}: {content}')

            member_roles = [role.name for role in author.roles]
            if 'Infant' in member_roles and 'Toddler' not in member_roles and 'Parent' not in member_roles:
                await self.assign_role(message.guild, author, 'Toddler')
                await self.remove_role(message.guild, author, 'Infant')
                print(f'{author.name} said their first word! They are Toddler now!')
                await self.log_channel.send(
                    f'{author.mention} said their first word! They are Toddler now! {message.jump_url}')

        await self.process_commands(message)  # this is needed for the bot to process commands

    def is_message_empty(self, message):
        return len(message) == 0

    async def on_member_join(self, member):
        if member not in self.inactive_members:
            print(f'{member.name} joined the server')
            await self.log_channel.send(f'{member.mention} joined the server! Hello Baby!')
            await self.assign_role(member.guild, member, 'Infant')
