# DiscordGPT Bot
import os
import discord
import openai
import googleapiclient.discovery
import requests
from bs4 import BeautifulSoup

# you need to load environment variable in some manner, use dotenv for example or just add them to your environment

# Replace with your own API key and custom search engine ID
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_ENGINE_ID = os.getenv('GOOGLE_ENGINE_ID')

# Create a custom search client
service = googleapiclient.discovery.build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

# load environment variables
TOKEN = os.getenv('DISCORD_TOKEN')
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_ENDPOINT_URL")
openai.api_version = os.getenv("OPENAI_API_VERSION")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def generate_summary(context, text):
    prompt = f"User asked: {context} As response we got a long text, please provide a brief summary:\n{text}"

    response = openai.ChatCompletion.create(
        engine="gpt-4-32k",
        messages=[{"role": "system",
                   "content": "You are a summarizer AI, your task is to condense and summarize the important "
                              "points in text given to you, convert any imperial units to metric SI units. "
                              "Windspeed should be indicated in m/s"},
                  {"role": "user", "content": prompt}],
        temperature=0.95,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    summary = response['choices'][0]['message']['content']
    return summary


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    if client.user in message.mentions:
        print(f"Query: {message.content}")
        try:
            if message.content.index("!search"):
                print("Search requested: " + message.content[message.content.index("!search")+8:])
                # Extract the search query from the message
                query = message.content[message.content.index("!search")+8:]

                # Call the Google Custom Search API and retrieve the first result
                result = service.cse().list(
                    q=query,
                    cx=GOOGLE_ENGINE_ID,
                    num=1
                ).execute()

                # Extract the URL and title of the first search result
                if "items" in result:
                    url = result["items"][0]["link"]
                    resp = requests.get(url)
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    summary = generate_summary(query, soup.get_text())
                    print(f"Bot: {summary}\n{url}")
                    await message.channel.send(summary)
                else:
                    await message.channel.send("Sorry, I couldn't find anything.")
        except ValueError:
            print(f"Chat requested {message.content}")
            response = openai.ChatCompletion.create(
                engine="gpt-4-32k",
                messages=[{"role": "system",
                           "content": "You are a witty and creative AI assistant named Aino. You should always be brief "
                                      "and to the point but provide enough information."},
                          {"role": "user", "content": message.content}],
                temperature=0.95,
                max_tokens=1024,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            print(f"Bot: {response['choices'][0]['message']['content']}")
            await message.channel.send(response['choices'][0]['message']['content'])


client.run(TOKEN)
