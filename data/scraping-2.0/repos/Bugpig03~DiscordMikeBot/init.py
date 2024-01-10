#-------- LIBRARIES --------
import discord
import random
from discord.ext import commands
from discord.utils import get
import os
import re
import time as classictime
from datetime import datetime, time, timedelta
from dotenv import load_dotenv
import asyncio
import ffmpeg
import requests
from blagues_api import BlaguesAPI
import openai
from pytube import YouTube
from pydub import AudioSegment
from youtubesearchpython import VideosSearch
import json
from quart import Quart, render_template, request, redirect
from aiohttp import web
import threading

#-------- VARIABLES DECLARATIONS --------
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='*', intents=intents) # instance bot discord
app = Quart(__name__) #instant server web quart
scores = {}
listProfiles = {}
newProfile = {}
MESSAGE_TIME = time(10, 35)
queueMusic = []
currentTOP10 = []

mainTextChannel = 479958977472364555
mainVoiceChannel = 426760269205602304

currentAnecdote = "Aucune annecdote disponible sorry :("
currentMusic = ""
currentMusicQueue = []

#-------- PATH CONFIGURATION --------
MUSIC_DIR = 'C:\Projet\Python\music'
MUSIC_DIR_YT = 'C:\Projet\Python\YTmusic'
SECRET_JSON_DIR = r"C:\Projet\Python\DiscordMikeBot\secrets.json"
PROFILE_JSON_DIR = r"C:\Users\Administrateur\source\repos\Bugpig03\DiscordMikeBot\profiles.json"


