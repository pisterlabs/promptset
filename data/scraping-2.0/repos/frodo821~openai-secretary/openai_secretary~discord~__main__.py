from os.path import dirname, join
from openai_secretary.discord import OpenAIChatBot

with open(join(dirname(__file__), '..', '..', '.discord.secret')) as f:
  key = f.read().strip()

bot = OpenAIChatBot(key, response_ratio=0.9)

bot.start()
