import discord
from discord.ext.commands import slash_command
from discord.ext import commands
from discord.ext.commands import Cog

import os 
import regex as re
from dotenv import load_dotenv
import requests
import numpy as np
import pandas as pd

import cohere
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http import models as rest
from qdrant_client.models import Filter


load_dotenv()  # load all the variables from the env file

CHAT_HISTORY_PATH = '/content/drive/MyDrive/career/projects/hackathons/lablab-cohere-qdrant-hackathon/discord-chat-history.csv'
BASE_MESSAGE_URL = "https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
QDRANT_CLOUD_HOST = "19531f2c-0717-4706-ac90-bd8dd1a6b0cc.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_COLLECTION_NAME = 'discord'

co = cohere.Client(os.getenv('COHERE_API_KEY'))
qdrant_client = QdrantClient(
    host=QDRANT_CLOUD_HOST, 
    prefer_grpc=False,
    api_key=os.getenv('QDRANT_API_KEY'),
)
discord_client = discord.Client()
bot = discord.Bot()

def embed_text(text: list, model='multilingual-22-12'):
    """Generate text embeddings."""
    if type(text) is str:
        text = [text]
    embeddings = co.embed(text, model=model)
    vectors = [list(map(float, vector)) for vector in embeddings.embeddings]
    return vectors


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

# Define a search filter for author
ignore_author = "Fetcher"
author_filter = Filter(**{"must_not": [{"key": "author", "match": {"value": "Fetcher"}},
                                       {"key": "author", "match": {"value": "Findio"}},
                                       {"key": "author", "match": {"value": "Chatter"}},
                                       ],
                       "must": [{ "key": "word_count", "range": { "gte": 3 }}]
                       })

@bot.slash_command(name="fetch", description="Search for messages by embedding")
async def fetch(ctx, query: str, k_max=5):
    min_words = 20
    vectors = embed_text(query)
    for vector in vectors:
        response = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=vector,
            query_filter=author_filter,
            limit=k_max,
        )
    results = [record.payload for record in response]

    # def get_plain(content: str):
    #     plain = re.sub(r'\n', ' \n', content)
    #     plain = re.sub(r'(>.*\n|```(.|\n)*```|`?|\n)', '', plain.lower().strip())
    #     return plain

    if len(results) > 0:
        output = []
        # result_str = f'Search query: "{query}":\nSearch results:\n'
        # for result in results:
            # TODO: summarize by thread not single messages
            # if len(result['content'].split()) >= min_words:
            #     summary = co.summarize( 
            #         text=result['content'],
            #         model='summarize-xlarge', 
            #         length='medium',
            #         extractiveness='low',
            #         temperature=0.3,
            #         additional_command="to remember a conversation"
            #     ).summary
            # else: 
            #     summary = result['content']

            # result_message = result['content']
            # if len(result_message) > 100:
            #     result_message = result_message[:100] + '...'
                
            # result_str += f"""
            # * {result['author']} wrote [{result['created_at'][:16]}]: 
            # [{result_message}]({result['jump_url']}) 
            # {result['guild_id']}/{result['channel_id']}/{result['msg_id']}
            # """

        embed=discord.Embed(color=0x1eff00)
        for result in results:
            embed.add_field(name=f"{result['author']} at {result['created_at'][:16]}\n{result['channel_id']}.{result['msg_id']}", 
                            value=f"[{result['content'][:200]}...]({result['jump_url']})", 
                            inline=False)
        embed.set_footer(text="Use `/discuss` to message user on this topic.")

        # await ctx.respond(content=result_str)
        await ctx.respond(f':wave: Your search results for "{query}"', embed=embed)
    else:
        await ctx.respond("No matching messages found.")


@bot.slash_command(name="revise", description="Revise sentence for clarity")
async def revise(ctx, sentence: str):
    """Use generate API to revise sentence."""
    
    prompt = f"Give me a better version of the following sentence that is more concise and clear, in a polite, fun, yet professional tone: {sentence}" 
    response = co.generate(model='command-xlarge-beta',  
                        prompt = prompt,  
                        max_tokens=90,  
                        temperature=0.5,  
                        )
    revised = response.generations[0].text
    await bot.wait_until_ready()
    if revised:
        embed=discord.Embed(color=0x1eff00)
        embed.add_field(name="Original", value=sentence, inline=False)
        embed.add_field(name="Revised", value=revised, inline=False)
        await ctx.respond(":wave: Here you go.", embed=embed)
        # await ctx.respond(content=f"__Old__:\n{sentence}\n__Revised__: {revised}")
    else:
        await ctx.respond(content="No revision available.") 


@bot.slash_command(name="discuss", description="Start a conversation on a topic.")
async def start_convo(ctx: discord.ApplicationContext, user, id: str):
    try: 
        channel_id, msg_id = id.split('.')
        try: 
            # Get the message
            msg = await ctx.fetch_message(int(msg_id))
        except: 
            # Try to get the thread's parent channel
            msg = await ctx.fetch_message(int(channel_id))

        def get_plain_thread(content: str):
            plain = re.sub(r'\n', ' \n', content)
            plain = re.sub(r'(>.*\n|```(.|\n)*```|`?|\n)', '', plain.lower().strip())
            return plain
        
        plain_thread = get_plain_thread(msg.content) + ' '
        thread_summary = ''
        if msg.flags.has_thread:
            async for m in msg.thread.history(limit=100, oldest_first=True):
                formatted_content = m.content
                plain_thread += get_plain_thread(formatted_content) + ' '
        else: 
            pass

        thread_summary = co.summarize(
            text=plain_thread,
            model='summarize-xlarge', 
            length='medium',
            extractiveness='low',
            temperature=0.3,
            additional_command="to remember a conversation"
        ).summary
        
        embed=discord.Embed(color=0x1eff00)
        embed.add_field(name=f"Original message thread", value=f"[{msg.content[:200]}...]({msg.jump_url})", inline=False)
        embed.add_field(name=f"TL;DR", value=thread_summary, inline=False)
        await ctx.respond(f':wave: {user}, <@{ctx.author.id}> wants to chat with you about below.', embed=embed)
    except:
        await ctx.respond("No message found.")


@bot.slash_command(name="getthread", description="Summarize a thread or message.")
async def get_thread(ctx: discord.ApplicationContext, id):
    try: 
        channel_id, msg_id = id.split('.')
        try: 
            # Get the message
            msg = await ctx.fetch_message(int(msg_id))
        except: 
            # Try to get the thread's parent channel
            msg = await ctx.fetch_message(int(channel_id))

        def get_plain_thread(content: str):
            plain = re.sub(r'\n', ' \n', content)
            plain = re.sub(r'(>.*\n|```(.|\n)*```|`?|\n)', '', plain.lower().strip())
            return plain

        plain_thread = get_plain_thread(msg.content) + ' '
        if msg.flags.has_thread:
            async for m in msg.thread.history(limit=100, oldest_first=True):
                formatted_content = m.content
                plain_thread += get_plain_thread(formatted_content) + ' '
        await ctx.respond(plain_thread)
    except:
        await ctx.respond("No message found.")


@bot.slash_command(name="keyword_search", description="Search for messages containing a keyword")
async def search_messages(ctx: commands.Context, keyword: str):
    channel = ctx.channel
    messages = []
    async for message in channel.history(limit=100):
        if keyword in message.content:
            messages.append(f"{message.content}\nURL: {message.jump_url}")

    if messages:
        messages_str = "\n".join(messages)

        # 將訊息分割成多個部分，每個部分不超過 1000 字元
        message_parts = [messages_str[i:i+1000] for i in range(0, len(messages_str), 1000)]
        for part in message_parts:
            await ctx.send(f"Matching messages:\n{part}")
    else:
        await ctx.send(f"No messages found containing '{keyword}'")


@bot.slash_command(name="savehistory", description="Save chat history")
async def fetch(ctx: discord.ApplicationContext):

    def is_command(msg): 
        """Checking if the message is a command call"""
        if len(msg.content) == 0:
            return False
        elif msg.content.split()[0] == '_scan':
            return True
        else:
            return False

    data = []
    async for msg in ctx.channel.history(limit=10000, oldest_first=True): 
        # if msg.author != ctx.user:                        
        # if not is_command(msg):          
        # Get root message
        data.append({'content': msg.content,
                    'created_at': msg.created_at,
                    'author': msg.author.name,
                    'jump_url': msg.jump_url,
                    'author_id': msg.author.id,
                    'msg_id': msg.id,
                    'channel_id': msg.channel.id,
                    'guild_id': msg.guild.id,
                    })    
        # Get thread messages (if any)
        if msg.flags.has_thread:
            async for thread_msg in msg.thread.history(limit=100, oldest_first=True):
                data.append({'content': thread_msg.content,
                            'created_at': thread_msg.created_at,
                            'author': thread_msg.author.name,
                            'jump_url': thread_msg.jump_url,
                            'author_id': thread_msg.author.id,
                            'msg_id': thread_msg.id,
                            'channel_id': thread_msg.channel.id,
                            'guild_id': thread_msg.guild.id,
                            })
            # if len(data) == limit:
            #     break
        
    data = pd.DataFrame(data)
    data.to_csv(CHAT_HISTORY_PATH)
    await ctx.respond("Chat history saved!")
    print(f'Chat history saved to {CHAT_HISTORY_PATH}')


bot.run(os.getenv('DISCORD_TOKEN'))
