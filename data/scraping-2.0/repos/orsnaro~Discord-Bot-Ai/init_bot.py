"""
                          Coder : Omar
                          Version : v2.5.5B
                          version Date :  8 / 11 / 2023
                          Code Type : python | Discrod | BARD | GPT | HTTP | ASYNC
                          Title : Initialization of Discord Bot
                          Interpreter : cPython  v3.11.0 [Compiler : MSC v.1933 AMD64]
"""
import discord
from discord.ext import commands , tasks
import utils_bot as util
import asyncio as aio
import random
import random2
from bardapi import BardAsync , Bard , BardCookies , SESSION_HEADERS
from openai import AsyncOpenAI
import openai
from inspect import getmembers , isfunction
import aiohttp
import requests
from pyrandmeme2 import pyrandmeme2
from pyrandmeme2 import palestina_free
# from quote_async.quote import quote #TODO ( complete your quote lib fork and make it fully async )
from quote import quote
from random_word import RandomWords
from datetime import datetime
import re
import pytz
import asyncforismatic.asyncforismatic as foris
import logging
import contextlib
import os
import keys
import sys
# from bard_key_refresh import regenerate_cookie #TODO:
#------------------------------------------------------------------------------------------------------------------------------------------#
#USER MODULES
#------------------------------------------------------------------------------------------------------------------------------------------#
def init_gpt_session():
   #by default checks keys in sys. env variables check func docstring
   gpt = AsyncOpenAI(api_key= keys.openaiAPI_KEY, organization= keys.openaiAPI_ORG_ID) 
   return gpt   

gpt = init_gpt_session()
#------------------------------------------------------------------------------------------------------------------------------------------#
def init_bard_session () :
   # session = requests.Session()
   # session.headers = {
   #          "Host": "bard.google.com",
   #          "X-Same-Domain": "1",
   #          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
   #          "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
   #          "Origin": "https://bard.google.com",
   #          "Referer": "https://bard.google.com/",
   #      }
   # session.cookies.set("__Secure-1PSID", bardAPI_KEY)
   # bard = Bard(token=bardAPI_KEY , session=session, timeout=30)

   
   bard = BardAsync(token=keys.bardAPI_KEY) #add -> language= 'ar' to respond in arabic only (experimental)
   
   # while True:
   # 	try :
   # 		bard = BardAsync(token= bardAPI_KEY ) #add -> language= 'ar' to respond in arabic only (experimental)
   # 		break;

   # 	except Exception as e :
   # 		regenerate_cookie()
   # 		print ( e.__str__() + " TESTING NOTE : this is manual error message from init_bard_session() function")

   return bard

bard = init_bard_session()

# regarding mentions for all discrod objects : messages , users , rules .. etc : https://discord.com/developers/docs/reference#message-formatting
admins_room_id = 889999601350881390
wizy_voice_channel = 890209823898107904
memes_highlights_ch_id = 1137242260203909151
narols_island_wizard_channel_id = 1118953370510696498
testing_wizard_channel_id = 1133103993942462577
wizard_channels = (narols_island_wizard_channel_id , testing_wizard_channel_id )
wizard_bot_id = 1117540489365827594
chat_chill_ch_id = 889535812167938088
proxy_sites = ["https://Proxy.server:3128" ,"https://95.217.167.241:8080"]#first one is the only proxy server available for free in pythonanywhere for more servers: https://hidemyna.me/en/proxy-list/?type=s#list
default_help_msg = f"""
                   GPTEOUS HELP MESSAGE```
                   **I'M MIGHTY GPTEOUS !** the first GPT-Spirit in Narol's island Volcan guardian , Island's Master right hand  and the begining of Island's new ERA etcetera etcetera... I mean you get it am very special here  :man_mage:

                  ** :sparkles: __COMMAND GROUP 1: ASK , I shall Answer! __:sparkles:  **

                   :label:    Ask me any where in the Island and  I shall INDEED answer you

                  :label:    The question must start start with mentioning me e.g.( <@{wizard_bot_id}> )

                  :label:    if you want to speak with me more freely with no mentions/commands
                              just type anything in my channel <#{narols_island_wizard_channel_id}> and I shall respond !


                               ** :sparkles: __COMMAND GROUP 2: Wise Quotes & Deep memes __:sparkles:  **

                  :label:    to get random meme at any time use 'BoringWizard'
                  :label:    to get random quote at any time use 'wisewiz'

                              :inbox_tray: _Aditional Functionalities and SPELLS coming soon ..._:inbox_tray:

                   __COMANDS LIST__
                   ```fix
                   1. @WizardSpirit "your_question"
                   2. `wiz` "your_question"
                   3. `bard` "your_question"
                   4. `wizard` "your_question"
                   5. `wizardspirit` "your_question"
                   6. `~ <any_of_prev_CMDs>` "your_question"
                   7. `wisewiz`
                   8. `BoringWizard`
                   _(all of them is case INsensitive)_
                   ```

                  ```fix
                  **WARNING**: sometimes I won't respond this is mainly due to exceeding max embed char limit
                  i.e.(6000chars)
                  ```
                  ```fix


                   """
override_help_msgP1 = f"""

> ***MIGHTY GPTEOUS I AM:man_mage:! *** `The first GPT-Spirit in Narol's island.  Volcan guardian. Island Master's right hand. The begining of Island's new ERA etcetera etcetera... I mean you get it am very special here` :fire_hashira:

 :inbox_tray: __Invite ME:__ https://discord.com/api/oauth2/authorize?client_id=1117540489365827594&permissions=8&scope=bot%20applications.commands

:label:    Ask me any where in the Island and  I shall INDEED answer you

:label:    The question must start start with mentioning me e.g.( <@{wizard_bot_id}> ) or just `wizy` <:shyomen:1014210603969171526>

:label:    if you want to speak with me more freely with `chat-mode` instead of `single-prompt` mode
            just type anything in my channel <#{narols_island_wizard_channel_id}> and I shall respond! _(if my creds :key:  and :cookie:  still valid and fresh!)_

:label:    to get a random meme at any time use `BoringWizard` :ghost:

:label:    to get random quote at any time use `wisewiz` :man_mage:

:label:    to make me join a voice channel use `wizyjoin` :green_circle:  _(you must be inside it first)_

:label:    to make me leave a voice channel use `wizyleave` :o:


                                     :inbox_tray: ***PLUS Additional Functionalities and SPELLS coming soon ...*** :inbox_tray: \n\n\n
              \n

                   """
override_help_msgP2 = f"""



* __FULL COMANDS LIST & ALIASES__
                           
         1. Ask The wizard (GPT)
         ```fix
         • `wizyGPT` "your_question"
         • `wizy` "your_question"
         • `wizardspirit` "your_question"
         • `~ <any_of_prev_CMDs>` "your_question"
         ```

         2. Get a Wizardy Quote
         ```fix
         • wisewiz
         ```

         3. Get a Wizardy Meme
         ```fix
         • boringwizard
         ```

         4. Check Status & Latency
         ```fix
         • ping (gets your message latency)
         • wiz_ping (gets bots latency)
         ```

         5. Control Commands _(only specific roles are eligible to use)_
         ```fix
         • quotesz <new size> (defaulted to 200 chars and max  is ~5070 chars)
         • togglerandom (control activity of #memes-highlights channel: `pass nothin` toggles, `0` disable, `1` enable normal mode, `2+` enable special events mode )
         • wizyaimode (controls the AI model used in wizard chat channel. `ai_name` values : `gpt` or `bard`)
         ```

         6. Voice Activity
         ```fix
         • wizyjoin
         • wizyleave
         • wizyplay <url> (with no url he plays default wizy mmo chill track)
         • wizypause
         • wizyresume
         • wizystop
         ```
         
         7. Special
         ```fix
         • wizyawakened
         ```


                                                                    (ALL COMMANDS ARE CASE INsensitive :man_mage:!)\n\n
__for known issues/bugs and planned updates please check wizy's GitHub repo. So long Isalnder :man_mage: !__
                     * <https://github.com/orsnaro/Discord-Bot-Ai/tree/production-AWS> \n\n

                                                                           `END OF WIZARD HELP MESSAGE`
                  """

#------------------------------------------------------------------------------------------------------------------------------------------#
#NOTE: in order to go  away from on_ready() issues override Bot class and move all on ready to it's setup_hook()
class CustomBot(commands.Bot):

   async def setup_hook(self):
      self.is_auto_memequote_state = 1 if len(sys.argv) <= 1 else int(sys.argv[1]) #0 off | 1 on normal mode | 2 on special mode
      self.default_voice_channel: int = wizy_voice_channel
      self.wizy_chat_ch_ai_type: str = 'gpt'
      self.guilds_not_playing_timer: dict[discord.guild.id, int] = {}
      self.resume_chill_if_free.start()
      self.auto_memequote_sender_task.start()
      self.play_chill_loop.start()

   async def on_ready(self):
      #next line needed to enable slash commands (slash commands are type of interactions not ctx or message or normal command any more)
      await self.tree.sync()
      print(f"Bot info: \n (magic and dunder attrs. excluded) ")
      for attr in dir(bot.user):
         if  not (attr.startswith('__') or  attr.startswith('_')) :
            value = getattr(bot.user, attr)
            print(f'{attr}: {value}')
      print(f"\n\n Bot '{bot.user}' Sucessfully connected to Discord!\n\n")


   @tasks.loop(seconds= 5)
   async def  resume_chill_if_free(self):
      for guild in self.guilds:
         increment_val_sec = 5 
         if guild.id in self.guilds_not_playing_timer:
            # check happens once every 5 secs so increment every time by 5 secs
            self.guilds_not_playing_timer[guild.id] += increment_val_sec
         else:
            self.guilds_not_playing_timer[guild.id] = increment_val_sec

         if guild.voice_client != None and guild.voice_client.is_connected():
            if not guild.voice_client.is_playing():
               threshold_sec = 180 #3 minutes
               if self.guilds_not_playing_timer[guild.id] >= threshold_sec:
                  #if there is any user in channel besides wizy the bot play chill music else stay silent
                  connected_users_cnt = len( guild.voice_client.channel.members ) - 1
                  if connected_users_cnt >= 1 :
                     await guild.voice_client.channel.send("*3+ minutes of Silence:pleading_face: resuming* **MMO Chill Track** ...")
                     await util.play_chill_track(guild)
                  else:
                     #TESTING
                     print("\n\n\n\n\n\n TESTING########################## \n\n\n\n\n there is only the bot in voice channel: don't start track... \n\n\n\n\n\n######################\n\n\n\n")
                     #TESTING
                     
                  self.guilds_not_playing_timer[guild.id] = 0
            else :
               self.guilds_not_playing_timer[guild.id] = 0
         else:
            self.guilds_not_playing_timer[guild.id] = 0

   @resume_chill_if_free.before_loop
   async def wait_bot_ready(self):
      await self.wait_until_ready()


   @tasks.loop(hours=2)
   async def auto_memequote_sender_task(self):
      await util.send_rand_quote_meme(is_special= True if self.is_auto_memequote_state >= 2 else False)

   @auto_memequote_sender_task.before_loop
   async def before_start_auto_memequote_sender(self):
      #TESTING
      print(f"\n\n\n\n\n TESTING#####################   \n\n\n you auto_memequote_sender state is : {self.is_auto_memequote_state} \n\n\n\n ######################")
      #TESTING
      await self.wait_until_ready()

   async def toggle_auto_memequote_sender_state(self, state:int = None ) -> bool :
      
      if state is None: 
         if self.is_auto_memequote_state > 0 :
            self.auto_memequote_sender_task.cancel()
            self.is_auto_memequote_state = 0
         else:
            self.auto_memequote_sender_task.start()
            self.is_auto_memequote_state = 1
      elif state == 0:
         if self.auto_memequote_sender_task.is_running():
            self.auto_memequote_sender_task.cancel()
            self.is_auto_memequote_state = 0
      elif state == 1:
         if not self.auto_memequote_sender_task.is_running():
            self.auto_memequote_sender_task.start()
            self.is_auto_memequote_state = 1
      elif state == 2: #special eventof type: FREE Palestine!
            self.is_auto_memequote_state = 2
         
         
      return self.is_auto_memequote_state

   @tasks.loop(hours= 3)
   async def play_chill_loop(self, target_ch: discord.VoiceChannel= None):
      #when booting up bot make him join admin room (only for my server wizy home!)
      targetVchannel = self.get_channel(self.default_voice_channel) if target_ch == None else target_ch
      server = targetVchannel.guild

      if server.voice_client is not None :
         server.voice_client.disconnect()

      await targetVchannel.connect()

      if not server.voice_client.is_playing() :
         await util.play_chill_track(server)


   @play_chill_loop.before_loop
   async def before_play_chill(self):
      await self.wait_until_ready()

   async def stop_play_chill_loop(self):
      self.play_chill_loop.cancel()
   async def start_play_chill_loop(self):
      self.play_chill_loop.start()
#------------------------------------------------------------------------------------------------------------------------------------------#
   
bot = CustomBot(
                  command_prefix= ("~", '', ' '),
                  case_insensitive= True,
                  strip_after_prefix= True,
                  intents=discord.Intents.all(),
                  allowed_mentions= discord.AllowedMentions(everyone= False),
                  description= default_help_msg,
                  status= discord.Status.online,
                  activity= discord.Game("/help"),
                  help_command= None,
               )
#------------------------------------------------------------------------------------------------------------------------------------------#
def get_last_conv_id()  : ...  #TODO
#------------------------------------------------------------------------------------------------------------------------------------------#
def boot_bot() :
   log_std = open("std.log" , 'a') #logs all stderr and stdout and discord.py msgs
   log_discord = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='a')#logs only discord.py msgs
   if 'IS_PRODUTCION' in os.environ and os.environ['IS_PRODUCTION'] == '1' :
      with contextlib.redirect_stdout(log_std):
         with contextlib.redirect_stderr(log_std):
            bot.run(keys.Token_gpteousBot , log_handler= log_discord)#default logging level is info
   else :
      bot.run(keys.Token_gpteousBot , log_level= logging.DEBUG) #default handler is stdout , debug log level is more verbose!
#------------------------------------------------------------------------------------------------------------------------------------------#