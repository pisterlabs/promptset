import discord
from discord.ext import commands

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Define the intents
intents = discord.Intents.all()

bot = commands.Bot(command_prefix='!',intents=intents)

client = discord.Client(intents=intents)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. If the AI does not know the answer to a "
        "question, it truthfully says it does not know."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

@bot.event
async def on_ready():
    print("bot online")

@bot.event
async def on_member_join(member):
    guild = bot.get_guild() # guild id
    channel = guild.get_channel() # welcome channel id
    
    await channel.send(f"Hello {member.mention}!")
    await channel.send("Here's a joke for you:")
    
    joke = conversation.predict(input=f"Make a joke about the name {member} only type out the joke")
    joke = '\n'.join(joke.split('\n')[1:])
    await channel.send(joke)
    await channel.send("Any questions you want to ask? (Place a '$' infront when doing so)")
    
@bot.event
async def on_message(message):    
    if message.author.bot:
        return
    
    query = message.content[1:]

    if message.content.startswith('$'):
        await message.channel.send(conversation.predict(input=query))

bot.run('YOUR_KEY')
