import asyncio
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os
import nltk
import magic



import discord
from discord.ext import commands
from dotenv import load_dotenv
from discord.utils import get

load_dotenv()
train_key= os.getenv('TRAIN_KEY')
TOKEN = os.getenv('TRAIN_TOKEN')


intents=discord.Intents.all()
bot = commands.Bot(command_prefix = '!', description = "Hey, I'm here to assist the AiGoLearning server, how can I help?", intents=intents)
bot.remove_command("help")


loader = DirectoryLoader('QA',glob='**/*.txt')
docs = loader.load()

char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_texts = char_text_splitter.split_documents(docs)

openAI_embeddings = OpenAIEmbeddings(openai_api_key=train_key)
vStore = Chroma.from_documents(doc_texts, openAI_embeddings)
model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vStore)




@bot.command()
async def help(ctx):
    embed = discord.Embed(title="About",description="""
    Hello! This bot has been created for your convenience to answer questions in the training bot channel and it has been completely trained on our evaluation process and teaching process. If it recognizes a question, it will redirect you to the training bot channel so please ask the bot questions before you ask us, it knows just as much as us! Please be aware that context is NOT saved and you will need to reiterate the problem/idea for every message.
    """, color=0x1A4FCF, url="https://aigolearning.org")

    embed.set_author(name= ctx.author.display_name, icon_url = ctx.author.avatar)
    embed.set_thumbnail(url="https://aigolearning.org/wp-content/uploads/2020/08/newlogo.png")

    await ctx.send(embed=embed)





# @bot.command()
# async def ak(ctx):
#     async with ctx.typing():
#         asyncio.sleep(3)
#     await ctx.send(model.run(ctx.message.content.replace("!ak", "")))





@bot.event
async def on_ready():
    guild_count = 0

    for guild in bot.guilds:
        print(f"- {guild.id} (name: {guild.name})")
        guild_count = guild_count + 1

    print(f'{bot.user} is in ' + str(guild_count) + " guilds.")

    print(f'{bot.user} has connected to Discord!')


    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="for !help"))

    channel = bot.get_channel(1138104058545197096)

    await channel.send("The bot is now ready!")
@bot.event
async def on_connect():
    await bot.wait_until_ready()
    channel = bot.get_channel(1138104058545197096)

    await channel.send("The bot has reconnected! ")


@bot.event
async def on_disconnect():
    channel = bot.get_channel(1138104058545197096)

    await channel.send("The bot has disconnected.")

@bot.event
async def on_message(message):
    author = message.author
    content = message.content
    guild = message.guild
    training = discord.utils.find(lambda r: r.name == "Training Instructors", message.guild.roles)
    print(content)

    if author == bot.user:
        return
    
    if "?" in content and message.channel.id != 1136669326343143504 and not(training in author.roles):
        await message.channel.send(f"{author.mention} Seems like you have a question! Please go ahead and ask it in <#1136669326343143504> first and then ask it in general if your question cannot be answered.")
    
    if message.channel.id == 1136669326343143504 and not(training in author.roles):
        async with message.channel.typing():
            asyncio.sleep(3)
        await message.channel.send(model.run(content))
    


    await bot.process_commands(message)

# @bot.event
# async def on_member_join(member):
#     channel = bot.get_channel(572588229338071061)
#     await channel.send("Welcome to AiGoLearning " + member.mention + "!")

bot.run(TOKEN)