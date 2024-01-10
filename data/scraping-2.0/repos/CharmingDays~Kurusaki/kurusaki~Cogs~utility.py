import asyncio
import typing
import discord
from discord.ext.commands.context import Context
import openai
import os
from discord.ext import commands
from discord.ext.commands import Cog, command,Context




class Chatbot:
    def __init__(self, memory_limit=5):
        self.context = []
        self.memory_limit = memory_limit
        self.gpt_version = 'gpt-4-1106-preview'
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def remove_last(self):
        self.context.pop()

    def receive_message(self, user_message):
        version = "gpt-3.5-turbo"
        if "v4:" in user_message:
            version = self.gpt_version
            user_message = user_message[3:]
        self.context.append(('user', user_message[3:]))
        # Generate a response considering the context
        response = self.generate_response(version)
        self.context.append(('bot', response))
        self.trim_context()
        if version == "gpt-4-1106-preview":
            self.remove_last()
        return response
            
    def generate_response(self,version):
        # Join all parts of the conversation into a single string to prep for GPT-3
        conversation_history = "\n".join([f"{speaker}: {message}" for speaker, message in self.context])
        prompt = f"{conversation_history}\nbot:"
        gpt_response = openai.ChatCompletion.create(
            model = version,messages= [{
                    'role':'user','content': prompt
                }]
        )
        
        # Extract and return the text portion of the response
        response_text = gpt_response.choices[0].message.content.strip()
        if len(response_text) > 2000:
            response_list = [response_text[i:i+2000] for i in range(0, len(response_text), 2000)]
            return response_list
        
        return response_text

    def trim_context(self):
        # This function ensures conversation history doesn't exceed token limits.
        while self.get_context_length() > self.memory_limit:
            self.context.pop(0)

    def get_context_length(self):
        # Counts the total tokens in the current context.
        return sum(len(f"{speaker}: {message}") for speaker, message in self.context)


class Utility(Cog):
    def __init__(self,bot) -> None:
        self.bot:commands.Bot = bot
        self.feed_back_channel_id = 1105158289446142012
        self.utilityDb = {'users':[185181025104560128],'guild':[]}
        self.chatbot = Chatbot(memory_limit=2048)


    @Cog.listener('on_command')
    async def command_correction(self,ctx:Context):
        pass

    # async def cog_command_error(self, ctx: Context, error: commands.CommandError):
    #     if ctx.command.is_on_cooldown(ctx):
    #         cooldown_timer = ctx.command.get_cooldown_retry_after(ctx)
    #         return await ctx.send(f'Command is on cooldown for {cooldown_timer} more seconds')

    @commands.cooldown(rate=5,per=120,type=commands.BucketType.user)
    @command(name='save-quote')
    async def save_quote(self,ctx:Context,msgId:typing.Optional[int]):
        """
        Save a message as a quote for your server members to view later via `view-quote` command
        """
        try:
            message = await ctx.channel.fetch_message(msgId)
            #TODO  Update the message attributes and contents into database
        except Exception as error:
            if isinstance(error,discord.NotFound):
                return await ctx.send(f'Message ID {msgId} does not exist for channel {ctx.channel.mention}')
            if isinstance(error,discord.Forbidden):
                return await ctx.send(f"Bot does not have permission to access the message with ID {msgId}")
            if isinstance(error,discord.HTTPException):
                return await ctx.send("Something went wrong while trying to retrieve the message.\nPlease try again later.")



    @command(name='autoCorrect')
    async def command_auto_correct(self,ctx:Context,auto_type="guild"):
        guildId = ctx.guild.id
        if auto_type.lower() not in ['guild','user']:
            return await ctx.send(f"Option {auto_type} not found.\nUse option `guild` or `user`.")
        
        if auto_type.lower() == "user" and ctx.author.id not in self.utilityDb['users']:
            self.utilityDb['users'].append(ctx.author.id)
        
        if auto_type.lower() == "guild" and ctx.guild.id not in self.utilityDb['guilds']:
            self.utilityDb['guild'].append(guildId)

        if auto_type.lower() == "guild":
            self.utilityDb[auto_type].remove(guildId)
        else:
            self.utilityDb[auto_type].pop(ctx.author.id)


    @command(name='bugReport')
    async def bug_report(self,ctx,*, message):
        """
        Make a bug report to the developer
        {command_prefix}{command_name}
        {command_prefix}{command_name} The resume command isn't working
        NOTE: Try to include as much info as you can so :)
        """
        #TODO  add to database
        channel = self.bot.get_channel(self.feed_back_channel_id)
        await channel.send(f"`{ctx.author}({ctx.author.id}):` **{message}**")
        await ctx.send(f"Bug report has been sent, thank you for your feedback.!")


    @command(name='requestFeature')
    async def request_feature(self,ctx,*,message):
        """
        Make a feature request to the developer
        {command_prefix}{command_name}
        message: The message to the developer about the feature 
        {command_prefix}{command_name} An anime command for generating random anime titles
        """
        #TODO  add to database
        channel = self.bot.get_channel(self.feed_back_channel_id)
        await channel.send(f"`{ctx.author}({ctx.author.id}):` **{message}**")
        return await ctx.send("Feature request sent!. Thank you.")


    @commands.Cog.listener('on_message')
    async def chat_gpt_message(self,message:discord.Message):
        if message.author.bot:
            return
        if message.channel.id == 1130492632498442241:
            async with message.channel.typing():
                await asyncio.sleep(1)
            chat = self.chatbot.receive_message(message.content)
            if isinstance(chat,list):
                for index,reply in enumerate(chat):
                    if index == 0:
                        await message.reply(reply)
                    await message.channel.send(reply)
            return await message.reply(chat)


async def setup(bot):
    await bot.add_cog(Utility(bot))