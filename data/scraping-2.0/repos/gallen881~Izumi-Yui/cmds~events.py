from discord.ext import commands
import discord

import random
import yaml
import time
import openai

from core.classes import Cog_Extension
import core.function as function
import func.form_w as fw
import func.temp as ad
import func.modifyinput as mi
from cmds.talk import Talk as CmdsTalk

from chatterbot import ChatBot

openai.api_key = function.open_json('./data/config.json')['token']['openai']


class Events(Cog_Extension):

    
    time_stamp = {}


    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user or message.content.startswith('-') or message.author.bot:
            return

        data = function.open_json('./data/www.json')
        if message.channel.id not in data['noww_id']:
            if message.content.endswith('w'):
                rd = random.randrange(10000)
                if rd <= -1:
                    await message.channel.send(fw.form_w('2000'))
                elif rd <= 4130:
                    await message.channel.send(fw.form_w('random'))


        data = function.open_json('./data/synchronous_channel.json')
        if str(message.channel.id) in data.keys():
            for c in data[str(message.channel.id)]:
                ch = self.bot.get_channel(c)
                if message.content != '':
                    await ch.send(f'*{message.author.name}#{message.author.discriminator} sent* **{message.content}** *(from {message.channel})*')

                for attachment in message.attachments:
                    await ch.send(f'*{message.author.name}#{message.author.discriminator} sent (from {message.channel})*')
                    await ch.send(attachment)


        if mi.IsZhInputs(message.content):
            zh = mi.ToZH(message.content)
            await message.reply(zh)
            await message.add_reaction('<:keyboard:854350638918008913>')
            function.print_detail(memo='INFO', user=message.author, guild=message.guild, channel=message.channel, obj=f'Traslate "{message.content}" to "{zh}"')


        data = function.open_json('./data/talk.json')
        if str(message.channel.id) in data.keys():
            async with message.channel.typing():
                if data[str(message.channel.id)] != 'openai':
                    try:
                        responce = CmdsTalk.chatbot[data[str(message.channel.id)]].get_response(message.content)
                    except:
                        CmdsTalk.chatbot[data[str(message.channel.id)]] = ChatBot(data[str(message.channel.id)], database_uri=f'sqlite:///data/{data[str(message.channel.id)]}.db')
                        responce = CmdsTalk.chatbot[data[str(message.channel.id)]].get_response(message.content)
                        function.print_detail(memo='WARN', user=message.author, guild=message.guild, channel=message.channel, obj=f'"{data[str(message.channel.id)]}" not found, create a new one')
                else:
                    responce = openai.Completion.create(model='text-davinci-003', prompt=message.content, temperature=0.5, max_tokens=60, top_p=0.3, frequency_penalty=0.5, presence_penalty=0.0, echo=True)['choices'][0]['text']
                await message.channel.send(responce)
                function.print_detail(memo='INFO',user=message.author, guild=message.guild, channel=message.channel, obj=f'"{message.content}" bot replied "{responce}"')


        data = function.open_json('./data/listen.json')
        if message.channel.id in data[str(message.guild.id)]:
            path = f'./chatterbot/chatterbot_corpus/data/local/{str(message.channel.id)}.yml'
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    yml = yaml.safe_load(file)
                    file.close()
                existing = True
            except:
                yml = {'categories': [str(message.channel.id)], 'conversations': []}
                existing = False



            if message.content != '':
                try:
                    self.time_stamp[str(message.channel.id)]
                except:
                    self.time_stamp[str(message.channel.id)] = 0

                now = time.time()
                if now - self.time_stamp[str(message.channel.id)] > 1800:
                    yml['conversations'].append([message.content])
                else:
                    try:
                        yml['conversations'][-1].append(message.content)
                    except:
                        yml['conversations'].append([message.content])

            self.time_stamp[str(message.channel.id)] = now

            with open(path, 'w', encoding='utf-8') as file:
                yaml.dump(yml, file, allow_unicode=True)
                file.close()

            if existing:
                function.print_detail(memo='INFO', user=message.author, guild=message.guild, channel=message.channel, obj=f'Add "{message.content}" to an existing conversation')
            else:
                function.print_detail(memo='INFO', user=message.author, guild=message.guild, channel=message.channel, obj=f'Add "{message.content}" to a new conversation')


    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id != self.bot.user.id:
            if payload.emoji.name == '\u274C' and payload.message_id == ad.url_data.msg.id:
                await ad.url_data.msg.delete()
                urls = function.open_json('./data/urls.json')
                urls['pinterest'].remove(ad.url_data.url)
                urls['nopinterest'].append(ad.url_data.url)
                function.write_json('./data/urls.json', urls)

                await self.bot.get_channel(payload.channel_id).send('Deleted picture successfully')

                function.print_detail(memo='INFO',user=payload.member, guild=payload.member.guild, channel=self.bot.get_channel(payload.channel_id), obj=f'{payload.member} deleted {ad.url_data.url} successfully')
         

async def setup(bot):
    await bot.add_cog(Events(bot))