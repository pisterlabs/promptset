import disnake
from disnake.ext import commands
import asyncio
import openai

import configs.private_config as private_config
import configs.public_config as public_config

import helpers.helpers as helpers
import helpers.database_logger as database_logger

from bots.music_instance import MusicBotInstance, Interaction


class MusicBotLeader(MusicBotInstance):
    instances = None
    chatgpt_messages = None

    def __init__(self, name, token, process_pool):
        super().__init__(name, token, process_pool)
        self.instances = []
        self.instances.append(self)
        self.instance_count = 0
        self.chatgpt_messages = {}
        openai.api_key = private_config.openai_api_key

        @self.bot.event
        async def on_voice_state_update(member, before: disnake.VoiceState, after: disnake.VoiceState):
            await self.on_voice_event(member, before, after)

            if await self.unmute_clients(member, before, after):
                return

            if member.guild.get_member(private_config.bot_ids["moderate"]) == None:
                if not after.channel:
                    if await helpers.check_admin_kick(member):
                        return

        @self.bot.event
        async def on_message(message):
            if await self.check_gpt_interaction(message):
                return

            if not message.guild:
                if helpers.is_supreme_being(message.author):
                    await message.reply(public_config.on_message_supreme_being)
                return

            if self.bot.get_user(private_config.bot_ids["moderate"]) == None:
                await self.check_message_content(message)
            await helpers.check_mentions(message, self.bot)

        @ self.bot.slash_command(dm_permission=False, description="Plays a song from youtube (paste URL or type a query)", aliases="p")
        async def play(inter: disnake.AppCmdInter,
                       query: str = commands.Param(description='Type a query or paste youtube URL')):
            await inter.response.defer()

            if not inter.author.voice or not inter.author.voice.channel:
                return await inter.send("You are not in voice channel")
            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                assigned_instance = await self.get_available_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no available bots, you can get more music bots in discord.gg/nazarick")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.play(new_inter, query)

        @ self.bot.slash_command(dm_permission=False, description="Plays anime radio or custom online radio")
        async def radio(inter: disnake.AppCmdInter,
                        url: str = commands.Param(default=public_config.radio_url, description="URL of online radio (mp3 player)")):
            await inter.response.defer()

            if not inter.author.voice or not inter.author.voice.channel:
                return await inter.send("You are not in voice channel")
            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                assigned_instance = await self.get_available_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no available bots, you can get more music bots in discord.gg/nazarick")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.play(new_inter, url, radio=True)

        @ self.bot.slash_command(dm_permission=False, description="Plays a song from youtube (paste URL or type a query) at position #1 in the queue", aliases="p")
        async def playnow(inter: disnake.AppCmdInter,
                          query: str = commands.Param(description='Type a query or paste youtube URL')):
            await inter.response.defer()

            if not inter.author.voice or not inter.author.voice.channel:
                return await inter.send("You are not in voice channel")
            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                assigned_instance = await self.get_available_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no available bots, you can get more music bots in discord.gg/nazarick")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.play(new_inter, query, playnow=True)

        @ self.bot.slash_command(dm_permission=False, description="Pauses/resumes player")
        async def pause(inter: disnake.AppCmdInter):
            await inter.response.defer()

            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no bots in your voice channel")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.pause(new_inter)

        @ self.bot.slash_command(dm_permission=False, description="Repeats current song")
        async def repeat(inter: disnake.AppCmdInter):
            await inter.response.defer()

            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no bots in your voice channel")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.repeat(new_inter)

        @ self.bot.slash_command(dm_permission=False, description="Clears queue and disconnects bot")
        async def stop(inter: disnake.AppCmdInter):
            await inter.response.defer()

            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no bots in your voice channel")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.stop(new_inter)

        @ self.bot.slash_command(dm_permission=False, description="Skips current song")
        async def skip(inter: disnake.AppCmdInter):
            await inter.response.defer()

            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no bots in your voice channel")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.skip(new_inter)

        @ self.bot.slash_command(dm_permission=False, description="Shows current queue")
        async def queue(inter: disnake.AppCmdInter):
            await inter.response.defer()

            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no bots in your voice channel")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.queue(new_inter)

        @ self.bot.slash_command(dm_permission=False, description="Removes last added song from queue")
        async def wrong(inter: disnake.AppCmdInter):
            await inter.response.defer()

            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no bots in your voice channel")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.wrong(new_inter)

        @ self.bot.slash_command(dm_permission=False, description="Shuffles current queue")
        async def shuffle(inter: disnake.AppCmdInter):
            await inter.response.defer()

            assigned_instance = await self.get_playing_instance(inter)
            if not assigned_instance:
                return await inter.send("There are no bots in your voice channel")
            new_inter = Interaction(assigned_instance.bot, inter)
            await assigned_instance.shuffle(new_inter)

        @ self.bot.slash_command(description="Allows to use ChatGPT")
        async def gpt(inter: disnake.AppCmdInter,
                      message: str = commands.Param(description="Type a query to get ChatGPT's reply")):
            await inter.response.defer()

            new_inter = Interaction(self.bot, inter)
            asyncio.create_task(self.gpt_helper(new_inter, message))

        @ self.bot.slash_command(description="Clears chat history with ChatGPT (it will forget all your messages)")
        async def gpt_clear(inter: disnake.AppCmdInter):
            await inter.response.defer()

            self.chatgpt_messages[inter.author.id] = []
            await inter.send("Done!", delete_after=5)

        @ self.bot.slash_command(description="Reviews list of commands")
        async def help(inter: disnake.AppCmdInter):
            await inter.response.defer()
            await inter.send(embed=disnake.Embed(color=0, description=self.help()))

    def add_instance(self, bot):
        self.instances.append(bot)


# *_______OnVoiceStateUpdate_________________________________________________________________________________________________________________________________________________________________________________________

    async def unmute_clients(self, member, before: disnake.VoiceState, after: disnake.VoiceState):
        if member.guild.get_member(private_config.bot_ids["moderate"]) != None:
            return False

        if after.channel:
            await helpers.unmute_bots(member)
            await helpers.unmute_admin(member)
            return True
        return False

# *_______OnMessage_________________________________________________________________________________________________________________________________________________________________________________________

    async def check_message_content(self, message) -> bool:
        if "discord.gg" in message.content.lower():
            if hasattr(message.author, "guild"):
                if not await helpers.is_admin(message.author):
                    await helpers.try_function(message.delete, True)
                    await helpers.try_function(message.author.send, True, f"Do NOT try to invite anyone to another servers {public_config.emojis['banned']}")
            else:
                await helpers.try_function(message.delete, True)
            return True
        return False

    async def check_gpt_interaction(self, message) -> bool:
        if message.author.bot:
            return False
        if not message.guild:
            inter = Interaction(self.bot, message)
            inter.orig_inter = None
            inter.message = message
            asyncio.create_task(self.gpt_helper(inter, message.content))
            return True
        if message.reference:
            ff, replied_message = await helpers.try_function(message.channel.fetch_message, True, message.reference.message_id)
            if not ff:
                return False
            if message.author.id not in self.chatgpt_messages or replied_message.author.id != self.bot.user.id:
                return False
            try:
                if replied_message.content[10:100] in self.chatgpt_messages[message.author.id][-1]["content"]:
                    inter = Interaction(self.bot, message)
                    inter.orig_inter = None
                    inter.message = message
                    asyncio.create_task(self.gpt_helper(inter, message.content))
                    return True
            except Exception as err:
                print(err)
                pass
        return False

# *______InstanceRelated____________________________________________________________________________________________________________________________________________________________________________________

    async def get_available_instance(self, inter):
        guild_id = inter.guild.id
        for instance in self.instances:
            if instance.contains_in_guild(guild_id) and instance.available(guild_id):
                # print("Returned fair instance")
                return instance
        for instance in self.instances:
            if instance.contains_in_guild(guild_id) and instance.check_timeout(guild_id):
                # print("Returned fair instance from timeout")
                return instance
        return None

    async def find_instance(self, inter):
        guild = inter.guild
        for instance in self.instances:
            if guild in instance.guilds:
                voice = instance.bot.get_guild(inter.guild.id).voice_client
                if voice and voice.channel == inter.author.voice.channel:
                    return instance
        for instance in self.instances:
            if guild in instance.guilds:
                voice = instance.bot.get_guild(inter.guild.id).voice_client
                if not voice or not voice.is_connected() or helpers.get_members_cont(voice.channel.members) == 1:
                    return instance
        if not await helpers.is_admin(inter.author):
            return None
        for instance in self.instances:
            if guild in instance.guilds:
                return instance

    async def get_playing_instance(self, inter):
        guild_id = inter.guild.id
        author_vc = None
        if inter.author.voice:
            author_vc = inter.author.voice.channel
        else:
            return None
        for instance in self.instances:
            if instance.contains_in_guild(guild_id) and instance.current_voice_channel(guild_id) == author_vc:
                return instance
        return None

# *______SlashCommands______________________________________________________________________________________________________________________________________________________________________________________

    async def gpt_helper(self, inter, message):
        if inter.author.id not in self.chatgpt_messages:
            self.chatgpt_messages[inter.author.id] = []
        messages_list = self.chatgpt_messages[inter.author.id]
        messages_list.append({"role": "user", "content": message})

        while True:
            try:
                response = await self.run_in_process(openai.ChatCompletion.create, model="gpt-3.5-turbo", messages=messages_list)
                response = response.choices[0].message.content
                break
            except Exception as e:
                print(e)
                if len(messages_list) > 1:
                    messages_list = messages_list[2:]

        chunks = helpers.split_into_chunks(response)

        if inter.orig_inter:
            await inter.orig_inter.edit_original_response(chunks[0])
        else:
            await inter.message.reply(chunks[0])
        for i in range(1, len(chunks)):
            await inter.text_channel.send(chunks[i])
        messages_list.append({"role": "assistant", "content": response})
        await database_logger.gpt(inter.author, [message, response])

    def help(self):
        ans = "Type **/play** to order a song (use URL from YT or just type the song's name)\n"
        ans += "Type **/stop** to stop playback\n"
        ans += "Type **/skip** to skip current track\n"
        ans += "Type **/queue** to print current queue\n"
        ans += "Type **/shuffle** to shuffle tracks in the queue\n"
        ans += "Type **/wrong** to remove last added track\n"
        ans += "Type **/repeat** to toogle repeat mode for current track\n"
        ans += "Type **/pause** to pause/resume playback\n"
        ans += "Type **/playnow** to order a song at pos #1 in the queue\n"
        ans += "Type **/radio** to play online radio (by default plays ANISON.FM)"
        return ans
