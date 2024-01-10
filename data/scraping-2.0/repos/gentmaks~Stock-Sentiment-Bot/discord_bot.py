import discord
from dotenv import load_dotenv
import os
import tweet_fetcher
import stock_graphs
import openai_classifier

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
client = discord.Client()


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    msg = message.content.split(" ")
    print(msg)
    if msg[0] == '!crypto-sentiment':
        await crypto_sentiment_analyze(message, msg[1])
    if msg[0] == '!dump-tweets':
        if len(msg) == 3:
            await dump_tweets(message, msg[1], int(msg[2]))
        else:
            await dump_tweets(message, msg[1], 10)
    if msg[0] == '!graph-stock':
        if len(msg) == 4:
            await graph_discord_stock(message, msg[1], msg[2], msg[3])
        else:
            await graph_discord_stock(message, msg[1], '1d', '5m')


async def crypto_sentiment_analyze(message, keyword):
    await message.channel.send("Stock sentiment info for '" + keyword + "' coming!")
    await message.channel.send(file=discord.File('doge.png'))
    tweets = tweet_fetcher.clean_tweets(tweet_fetcher.get_tweets(keyword, 10))
    positive = 0
    negative = 0
    for tweet in tweets:
        sentiment = openai_classifier.make_single_request(tweet)
        if sentiment == "Positive":
            positive += 1
        elif sentiment == "Negative":
            negative += 1
    await message.channel.send("Out of 10 tweets posted in the past day: \n" +
                               str(positive) + " appear to be Positive \n" +
                               str(negative) + " appear to be Negative \n")


async def dump_tweets(message, keyword, number_of_tweets):
    tw = tweet_fetcher.clean_tweets(tweet_fetcher.get_tweets(keyword, number_of_tweets))
    msg = ""
    await message.channel.send("Here are " + str(number_of_tweets)
                               + " tweets that include " + keyword)
    for chunk in chunks(tw, 10):
        for tweet in chunk:
            msg += tweet + " \n\n"
        await message.channel.send(msg)
        msg = ""


async def graph_discord_stock(message, symbol, period, interval):
    stock_graphs.graph_stock(symbol, period, interval)
    await message.channel.send(file=discord.File('plot.png'))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


client.run(TOKEN)
