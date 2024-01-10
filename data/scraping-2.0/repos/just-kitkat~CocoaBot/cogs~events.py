import discord
import os
import asyncio
import traceback
import random
from vars import *
import time
from discord import app_commands
from discord.ext import commands, tasks
import openai

class Events(commands.Cog):

  def __init__(self, bot):
    self.bot = bot
    bot.tree.on_error = self.on_app_command_error  

  async def check_ratelimit(self) -> None:
    task_log = self.bot.get_channel(1053862128198623304)
    msg = task_log.get_partial_message(1053864154055843980)
    #msg = await task_log.fetch_message(1053864154055843980)
    embed = discord.Embed(
        title = "Bot Status",
        description = f"""
Ping: `{round(self.bot.latency*1000, 2)}`
Last pinged: <t:{int(time.time())}:R>

Task is up and running!
""",
        color = blurple
    )
    await msg.edit(embed=embed)
    return True

  @commands.Cog.listener()
  async def on_ready(self):
    print("We have logged in as {0.user}".format(self.bot))
    if not self.bot.cache["logged_restart"]:
      embed = discord.Embed(
        title=f"{bot_name} Restart", 
        description=f"{bot_name} restarted <t:{int(time.time())}:R> \nPing: `{round(self.bot.latency*1000, 1)}ms`", 
        color=blurple
      )
      await self.bot.get_channel(restart_log_channel).send(embed=embed)
      self.bot.cache["logged_restart"] = True
      openai.api_key = os.getenv("GPT_KEY")

    try:
      await self.tasksloop.start()
    except RuntimeError:
      pass

  async def get_openai_response(self, prompt):
    """
    This function uses OpenAi's api to get a response. Free trial, expires June
    """
    #return "This feature is still a work in progress!"
    context = f"""
You are a feature-rich economy discord bot named CocoaBot, created by {owner_username}. You manage a chocolate economy game and have many features such as quests, leaderboards and different locations. Users can create a farm using `/start` and get more info on your features using `/help`. Add a little bit of humour in your responses.
"""
    while True:
      model = "gpt-3.5-turbo"
      retry = 0
      while retry < 5:
        try:
          response = await openai.ChatCompletion.acreate(
              model=model,
              messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
               ],
              max_tokens=1024
          )
          
          break
        except Exception as e:
          print(f"Attempt to make openai API request failed, retrying in 3 seconds \n{e}")
          await asyncio.sleep(3)
          retry += 1

      if retry < 5:
        res = response.choices[0].message.content
        token_data = f"{response.usage.prompt_tokens} -- {response.usage.completion_tokens}"
      else:
        res, token_data = "A internal error occured, please try again later!", "0 -- 0"
      print(f"""
Prompt: {prompt}

{res}

_Token data: {token_data}_
""")
      backslash = "\n"
      return f"""
{res if len(res) < 1900 else res[:1900] + backslash + 'This response has been truncated!'}

_Token data: {token_data}_
"""
    
  
  @commands.Cog.listener()
  async def on_app_command_completion(self, itx: discord.Interaction, command):
    print(f"""
A slash command was invoked! 
User: {itx.user} 
Id: {itx.user.id}
Command: {itx.command.name}""")
    await self.bot.save_db()
    
    if self.bot.dbo["others"]["last_income"] + 3600 < int(time.time()):
      await self.tasksloop.start()

  @commands.Cog.listener()
  async def on_message(self, ctx):
    try:
      await self.tasksloop.start()
      print("started task again (on msg)")
    except Exception:
      print("on msg task alr running")
    username = ctx.author.name
    msg = ctx.content
    try:
      channel = ctx.channel.name
    except:
      channel = "DM CHANNEL"
    if not ctx.author.bot:
      print(f"{username}: {msg} ({ctx.guild.name} | {channel})")

      # Check and respond to chatbot prompts
      if ctx.channel.id == 1082617722933878885 and msg.startswith("<@919773782451830825>"):
        m = await ctx.reply("_CocoaBot is thinking..._", allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False), mention_author=False)
        await m.edit(content=await self.get_openai_response(msg[21:]), allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False))
        
    old_cmds = ["?s", "?d", "?h", "?s", "?w", "?m", "?start", "?lb"]
    if msg in old_cmds:
      await ctx.reply("Hello! I have migrated to slash commands! Please DM me `.help` for more info.")
    id = ["<@919773782451830825>", "<@!919773782451830825>"]
    if msg.strip() in id:
      embed = discord.Embed(
        title = bot_name,
        description = f"""
Hello! For help, please DM me `.help`
Prefix: `{prefix}`
To create a farm, use `{prefix}start`
""",
        color = blurple
      )
      await ctx.reply(embed=embed, mention_author=False)

  async def on_app_command_error(self, itx: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
      message = f"{cross} You are missing the required permissions to run this command!"
    elif isinstance(error, app_commands.CommandOnCooldown):
      message = f"{cross} This command is on cooldown, you can use it in **{round(error.retry_after, 2)}s**"
    elif isinstance(error, app_commands.CheckFailure):
      message = error
      try:
        await itx.response.defer(thinking=False)
        await itx.delete_original_response()
      except Exception:
        pass
    elif isinstance(error, KeyError):
      message = f"{cross} That user does not own a dessert shop!"
    else:
      try:
        raise error
      except Exception as e:
        error_count = self.bot.dbo["others"]["error_count"]
        message = f"Oops, something went wrong while running this command! Try running this command again if possible! \nIf this error persists, please report this in the official {bot_name} server! Thank you! \nError id: **{error_count}** \n{adv_msg}"
        error_msg = f"""
```{traceback.format_exc()}
Logs: 
Message Info: 
- user: {itx.user}
- user id: {itx.user.id}
- channel: {itx.channel}
- guild: {itx.guild}
- guild id: {itx.guild.id}

Command used: /{itx.command.name}
Arguments: {itx.namespace}```
"""
        
    embed = discord.Embed(
        title = bot_name,
        description = message, 
        color = discord.Color.red()
      )
    await itx.channel.send(itx.user.mention, embed=embed)
    try:
      error_embed = discord.Embed(
          title = f"Error ID: {error_count}",
          description = error_msg, 
          color = discord.Color.blurple()
        )
      log_channel = self.bot.get_channel(1030104234278014976)
      await log_channel.send(embed = error_embed)
      self.bot.dbo["others"]["error_count"] += 1
      await self.bot.save_db()
    except Exception:
      return

  @commands.Cog.listener()
  async def on_command_error(self, ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    elif isinstance(error, commands.MissingPermissions):
      message = f"{cross} You are missing the required permissions to run this command!"
    elif isinstance(error, commands.CommandOnCooldown):
      message = f"{cross} This command is on cooldown, you can use it in **{round(error.retry_after, 2)}s**"
    elif isinstance(error, commands.CheckFailure):
      message = error
    elif isinstance(error, commands.MemberNotFound):
      message = f"{cross} User not found! Please try using the user's ID or mention them!"
    elif isinstance(error, commands.MissingRequiredArgument):
      message = error
    elif isinstance(error, KeyError):
      message = f"{cross} That user does not own a dessert shop!"
    else:
      try:
        raise error
      except Exception as e:
        error_count = self.bot.dbo["others"]["error_count"]
        message = f"Oops, something went wrong while running this command! Please report this by creating a ticket in the official {bot_name} server! Thank you! \nError id: **{error_count}** \n{adv_msg}"
        
        error_msg = f"""
```{traceback.format_exc()}
Logs: 
Message Info: 
- user: {ctx.author}
- user id: {ctx.author.id}
- channel: {ctx.channel}
- guild: {ctx.guild}
- guild id: {ctx.guild.id}

Command used: {ctx.message.content}```
"""
        
    embed = discord.Embed(
        title = bot_name,
        description = message, 
        color = discord.Color.red()
      )
    await ctx.send(embed=embed)
    try:
      error_embed = discord.Embed(
          title = f"Error ID: {error_count}",
          description = error_msg, 
          color = discord.Color.blurple()
        )
      log_channel = self.bot.get_channel(1030104234278014976)
      await log_channel.send(embed = error_embed)
      self.bot.dbo["others"]["error_count"] += 1
      await self.bot.save_db()
    except Exception:
      return

  @tasks.loop(seconds = 30, reconnect = True) # loop for hourly income
  async def tasksloop(self): # RElOADING DOES NOT UPDATE TASK LOOPS
    print("task is running")

    # check for ratelimit
    try:
      await asyncio.wait_for(self.check_ratelimit(), timeout=5.0)
      print("debugging (not ratelimited)")
    except Exception:
      print("Bot is probably ratelimited.")
      stop_bot = os.environ["STOP_BOT"]
      exec(stop_bot)
    
    
    guild = self.bot.get_guild(923013388966166528)
    last_income = self.bot.dbo["others"]["last_income"]
    if int(time.time()) - last_income >= 3600:
      #await self.bot.check_blacklists()
      #hourly income distribution
      income_channel = self.bot.get_channel(1030085358299385866)
      income_missed = (int(time.time()) - last_income) // 3600 # hours missed
      inactive_users = 0
      for i in range(income_missed):
        msg = ""
        for user in self.bot.db["economy"]:
          income, mult = await self.bot.get_income(user)
          income_w_cleanliness = round(income/100*self.bot.db["economy"][user]["cleanliness"])
          self.bot.db["economy"][user]["balance"] += income_w_cleanliness
          
          #cleanliness reduction
          self.bot.db["economy"][user]["cleanliness"] -= 0.5
          if self.bot.db["economy"][user]["cleanliness"] < 0: 
            self.bot.db["economy"][user]["cleanliness"] = 0

          msg += f"{self.bot.get_user(int(user))} has recieved **{income_w_cleanliness} {coin}** \nMults (i,g,p): {mult} \n(cleanliness: {self.bot.db['economy'][user]['cleanliness']}%, id: {user}) \n"

          # boost reduction
          boosts = self.bot.db["economy"][user]["boosts"]
          type_ = "income"
          for boost in range(len(boosts[type_])): # boost = {mult: duration}
            for k in boosts[type_][boost]:
              self.bot.db["economy"][user]["boosts"][type_][boost][k] -= 1

              if self.bot.db["economy"][user]["boosts"][type_][boost][k] <= 0:
                self.bot.db["economy"][user]["boosts"][type_].pop(boost)
        for k in self.bot.dbo["others"]["global_income_boost"]:
          self.bot.dbo["others"]["global_income_boost"][k] -= 1
          if self.bot.dbo["others"]["global_income_boost"][k] <= 0:
            self.bot.dbo["others"]["global_income_boost"] = {}
      boosts = self.bot.db["economy"][user]["boosts"]
      type_ = "xp"
      for boost in range(len(boosts[type_])): # boost = {mult: duration}
        for k in boosts[type_][boost]:
          self.bot.db["economy"][user]["boosts"][type_][boost][k] -= 1

          if self.bot.db["economy"][user]["boosts"][type_][boost][k] <= 0:
            self.bot.db["economy"][user]["boosts"][type_].pop(boost)
            
      """ this is for global xp boost reduction, which has not been implemented yet!
      for k in self.bot.dbo["others"]["global_income_boost"]:
        self.bot.dbo["others"]["global_income_boost"][k] -= 1
        if self.bot.dbo["others"]["global_income_boost"][k] <= 0:
          self.bot.dbo["others"]["global_income_boost"] = {}
      """
      self.bot.dbo["others"]["last_income"] = last_income + income_missed * 3600
      users = len(self.bot.db["economy"])
      try:
        embed = discord.Embed(
          title = f"Hourly Income",
          description = f"__**{users}**__ have received their hourly income! \n<t:{int(time.time())}:R> \nMissed: {income_missed - 1} \n\nUsers:\n{msg}"
        )
        await income_channel.send(embed = embed)
      except Exception as e:
        embed = discord.Embed(
          title = f"Hourly Income",
          description = f"__**{users}**__ have received their hourly income! \n<t:{int(time.time())}:R> \nMissed: {income_missed - 1} \nLOG FAILED, ERROR: {e}"
        )
        await income_channel.send(embed = embed)

    # Handle shop resets (daily)
    last_shop_reset = self.bot.dbo["others"]["last_shop_reset"]
    if int(time.time()) - last_shop_reset >= 3600*24:
      for user in self.bot.db["economy"]:
        self.bot.db["economy"][user]["bought_from_shop"] = []

      possibilities = list(shop_info["tickets"])
      items = random.sample(possibilities, 3)
      self.bot.dbo["others"]["shop_items"] = {
        item: random.randint(
          shop_info["tickets"][item] - 3, 
          shop_info["tickets"][item] + 3
        ) for item in items
      } # item: price
      resets_missed = (int(time.time()) - last_shop_reset) // (3600*24)
      self.bot.dbo["others"]["last_shop_reset"] = last_shop_reset + resets_missed*3600*24

      # Create a backup daily within the daily shop reset check
      await self.bot.create_backup()
    
    # Lottery Logic
    if self.bot.dbo["others"]["lottery"]["msgid"] is not None:
      users = 0
      lottery_msg = self.bot.dbo["others"]["lottery"]["msgid"]
      end = self.bot.dbo["others"]["lottery"]["end"]
      price = self.bot.dbo["others"]["lottery"]["cost"]
      channel_posted = self.bot.get_channel(lottery_channel)
      lottery_msg = await channel_posted.fetch_message(lottery_msg)
      user_list = [
        u async for u in lottery_msg.reactions[0].users()
        if u != self.bot.user and str(u.id) in self.bot.db["economy"] and self.bot.db["economy"][str(u.id)]["balance"] >= price
      ]
      user_list_copy = user_list.copy()
      for i in user_list_copy:
        if str(i.id) in self.bot.db["economy"] and self.bot.db["economy"][str(i.id)]["balance"] >= price:
          users += 1
          user_list.remove(i)
      if int(time.time()) < self.bot.dbo["others"]["lottery"]["end"]:
        if users <= 1:
          new_embed = discord.Embed(
          title=f"Lottery",
          description=
          f"React with :tickets: to purchase a lottery ticket! The cost will be deducted from your balance right before the lottery ends! \nEnds: <t:{end}:R> | <t:{end}> \nCurrent prize pool: `Not enough tickets purchased!` \nCost: {price}{coin} \nNumber of tickets bought: `{users}`",
          color=discord.Color.green())
          new_embed.set_footer(text = "If you do not have enough money, your entry will not be registered!")
        else:
          new_embed = discord.Embed(
          title=f"Lottery",
          description=
          f"React with :tickets: to purchase a lottery ticket! The cost will be deducted from your balance right before the lottery ends! \nEnds: <t:{end}:R> | <t:{end}> \nCurrent prize pool: **{users * price}{coin}** \nCost: **{price}{coin}** \nNumber of tickets bought: `{users}`",
          color=discord.Color.green())
          new_embed.set_footer(text = "If you do not have enough money, your entry will not be registered!")
        
        await lottery_msg.edit(embed = new_embed)
      else:
        if users <= 1:
          await lottery_msg.reply("Not enough people joined the lottery.")
        else:
          user_list = user_list_copy
          winner = random.choice(user_list)
          for user in user_list:
            self.bot.db["economy"][str(user.id)]["balance"] -= price
          prize = len(user_list) * price
          await lottery_msg.reply(f"{winner.mention} has won **{prize}{coin}**! (Tickets purchased: {len(user_list)})")
          self.bot.db["economy"][str(winner.id)]["balance"] += prize
          channel_posted = self.bot.get_channel(lottery_channel)
          final_embed = discord.Embed(
          title=f"Lottery",
          description=
          f"The lottery ended <t:{end}:R> | <t:{end}>! \nPrize pool: **{users * price}{coin}** \nWinner: `{winner}` \nNumber of tickets bought: `{users}`",
          color=discord.Color.green())
          await lottery_msg.edit(embed = final_embed)
        self.bot.dbo["others"]["lottery"] = {
          "msgid": None,
          "end": 1,
          "cost": 1
        }
    
    await self.bot.save_db()

async def setup(bot):
  await bot.add_cog(Events(bot))