import discord
from discord.ext import commands
import os
import random
import openai

defaultEmbedColor=discord.Color(0xe67e22)
green = discord.Color(0x00FF00)
red = discord.Color(0xFF0000)
openai.api_key = os.getenv("DALLE_key")

async def setStatus(self):
    theme = ""
    with open('/home/captain/boot/NTT/files/theme.txt', 'r') as t:
        theme = t.readlines()[0].strip()
    status = random.randint(1,3)
    response = ""
    match theme:
        case "default":
            match status:
                case 1:
                    await self.bot.change_presence(activity=discord.Game(name=str("with the bugs in my code")))
#                    response = openai.ChatCompletion.create(
#                            model="gpt-3.5-turbo",
#                            messages=[
#                                {"role": "system", "content": "You are a Discord status generator"},
#                                {"role": "user", "content": "Generate a short but witty discord status for a Bot that is doing nothing that starts with the word \"Playing\""}
#                            ]
#                    )
#                    response = response['choices'][0]['message']['content']
#                    response = response.split("\"")
#                    await self.bot.change_presence(activity=discord.Game(name=str(response[1].split("Playing")[1])))
#                    response = ""
                case 2:
                    await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=str("the birds chirp")))
#                    response = openai.ChatCompletion.create(
#                            model="gpt-3.5-turbo",
#                            messages=[
#                                {"role": "system", "content": "You are a Discord status generator"},
#                                {"role": "user", "content": "Generate a short but witty discord status for a Bot that is doing nothing that starts with the words \"Listening to\""}
#                            ]
#                    )
#                    response = response['choices'][0]['message']['content']
#                    response = response.split("\"")
#                    await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=str(response[1].split("Listening to")[1])))
#                    response = ""
                case 3:
                    await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=str("the day go by")))
#                    response = openai.ChatCompletion.create(
#                            model="gpt-3.5-turbo",
#                            messages=[
#                                {"role": "system", "content": "You are a Discord status generator"},
#                                {"role": "user", "content": "Generate a short but witty discord status for a Bot that is doing nothing that starts with the word \"Watching\""}
#                            ]
#                    )
#                    response = response['choices'][0]['message']['content']
#                    response = response.split("\"")
#                    await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=str(response[1].split("Watching")[1])))
#                    response = ""

        case "winter":
            match status:
                case 1:
                    await self.bot.change_presence(activity=discord.Game(name="with my new toys!"))
                case 2:
                    await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="carols!"))
                case 3:
                    await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="the snow fall!"))


class ThemesCog(commands.Cog):
    def __init__(self,bot):
        self.bot = bot

    async def changeStatus(self):
        await setStatus(self)
        #return

    @commands.Cog.listener()
    async def on_ready(self):
        print("Online!")
        await setStatus(self)
        print("Set status!")

    # Print code here
    @commands.command(name="theme", hidden=True)
    @commands.cooldown(1, 600, commands.BucketType.guild)
    @commands.has_permissions(administrator = True)
    async def theme(self, ctx, arg=None):
        theme = str(arg)
        match arg:
            case None:
                defaultEmbed = discord.Embed(color=defaultEmbedColor)
                dirs = ""
                for dir in os.listdir('/home/captain/boot/NTT/files/themes/'):
                    dirs+=str(dir)+'\n'
                defaultEmbed.add_field(name="Available themes:", value=dirs)
                await ctx.reply(embed=defaultEmbed)
            case "default":
                themesEmbed = discord.Embed(color=green,description=":white_check_mark: Done! Welcome to The Campfire!")
                with open('/home/captain/boot/NTT/files/themes/default/TheIcon.gif', 'rb') as icon:
                    await ctx.guild.edit(icon=icon.read())
                with open('/home/captain/boot/NTT/files/themes/default/TheCampfire.jpg', 'rb') as banner:
                    await ctx.guild.edit(banner=banner.read())
                with open('/home/captain/boot/NTT/files/themes/default/TheEntity.jpg', 'rb') as pfp:
                    await self.bot.user.edit(avatar=pfp.read())
                with open('/home/captain/boot/NTT/files/theme.txt', 'w') as theme:
                    theme.write("default")
                await ctx.reply(embed=themesEmbed)
            case "winter":
                themesEmbed = discord.Embed(color=green,description=":white_check_mark: Done! Happy Holidays!")
                with open('/home/captain/boot/NTT/files/themes/winter/XmasIcon.gif', 'rb') as icon:
                    await ctx.guild.edit(icon=icon.read())
                with open('/home/captain/boot/NTT/files/themes/winter/XmasCampfire.jpg', 'rb') as banner:
                    await ctx.guild.edit(banner=banner.read())
                with open('/home/captain/boot/NTT/files/themes/winter/XmasEntity.jpg', 'rb') as pfp:
                    await self.bot.user.edit(avatar=pfp.read())
                with open('/home/captain/boot/NTT/files/theme.txt', 'w') as theme:
                    theme.write("winter")

                await ctx.reply(embed=themesEmbed)
        await setStatus(self)
    
async def setup(bot):
	await bot.add_cog(ThemesCog(bot))
