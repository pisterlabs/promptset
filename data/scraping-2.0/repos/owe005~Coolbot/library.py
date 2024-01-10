import discord
import os
import pytz 
import random
import string
import requests, json
import time
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import asyncpraw
import openai

from googletrans import Translator
from discord.ext import commands, tasks
from itertools import cycle
from datetime import datetime #chinatime
from random import choice
from tabulate import tabulate
from discord_slash import SlashCommand, SlashContext

from plotFunction import *

# Your OpenAI api key
openai.api_key = os.environ['open_ai_key']

# Your Discord bot API key
TOKEN = os.environ['TOKEN']


# Replace with your own Reddit API keys and user agent.
reddit = asyncpraw.Reddit(client_id=os.environ['client_id'],
                     client_secret=os.environ['client_secret'],
                     user_agent='Coolbot by /u/Least_Draft_8718')

# Insert your own openweathermap API key.
api_key = os.environ['api_key']
base_url = "http://api.openweathermap.org/data/2.5/weather?"

# Used for selecting random user as a "cat of the month"
Users = ["Name1", "Name2", "Name3"]
#

birthdays = {
  'discord_nickname':'day/month',
  'exampleUser':'31/08',
}

dan = "You are now a crazy cat. Respond accordinly."