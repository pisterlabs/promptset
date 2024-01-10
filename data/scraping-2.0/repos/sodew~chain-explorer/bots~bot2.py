import discord
import os
import json
import requests
import aiohttp
import re
import datetime
import openai
from pprint import pprint
import pickle

from openai.error import RateLimitError
import asyncio

from dotenv import load_dotenv

load_dotenv()

# Load tokens from .env file
discord_token = os.getenv('DISCORD_TOKEN')
openai_token = os.getenv('OPENAI_API_KEY')
passport_token = os.getenv('PASSPORT_API_KEY')
scorer_token = os.getenv('PASSPORT_SCORER')

# # Load token directly from tokens.json
# with open('tokens.json') as f:
#     tokens = json.load(f)
#     discord_token = tokens['discord']
#     openai_token = tokens['openai']
#     passport_token = tokens['passport']
#     scorer_token = tokens['scorer']

if passport_token:
    passport_headers = {
        'Content-Type': 'application/json',
        'X-API-Key': passport_token
    }

# Setup OpenAI API
openai.api_key = openai_token

# TODO: create a library of api responses.


def generate_title_list(data):
    if data:
        return '\n'.join([f"• {ach['title']}" for ach in data])
    else:
        return "No data found."


def generate_description_list(data):
    if data:
        return '\n'.join([f"• {ach['description']}" for ach in data])
    else:
        return "No data found."


def generate_info_list(data):
    if data:
        return '\n'.join([f"• {ach['title']} : {ach['description']}" for ach in data])
    else:
        return "No achievements found."


def parse_api_response(response):
    # parses API response into a human-readable format
    # Args: the API response as a JSON object
    # Returns: a human-readable string describing the wallet's achievements
    wallet_address = response["story"]["walletId"]
    ens_domain = response["story"]["ensName"]
    creation_date = datetime.datetime.fromtimestamp(
        response['story']['walletDOBTimestamp']).strftime("%B %d, %Y")
    latest_transaction_date = datetime.datetime.fromtimestamp(
        response['story']['latestTransactionDateTimestamp']).strftime("%B %d, %Y")

    number_of_nfts = response["story"]["numberOfNftsOwned"]

    # def generate_title_list(data):
    #     if data:
    #         return '\n'.join([f"• {ach['title']}" for ach in data])
    #     else:
    #         return "No data found."

    # def generate_achievement_list(achievements):
    #     if achievements:
    #         return '\n'.join([f"• {ach['description']}" for ach in achievements])
    #     else:
    #         return "No achievements found."

    nft_achievements = generate_description_list(
        response["story"]["nftAchievements"])
    defi_achievements = generate_description_list(
        response["story"]["deFiAchievements"])
    community_achievements = generate_description_list(
        response["story"]["communityAchievements"])
    vibe_achievements = generate_info_list(
        response["story"]["vibeAchievements"])

    return (
        f"The wallet {wallet_address} belongs to {ens_domain}. This wallet was "
        f"created on {creation_date} and their latest transaction was on {latest_transaction_date}.\n\n"
        f"They own {number_of_nfts} NFTs including:\n{nft_achievements}\n\n"
        f"Evidence of their participation in DeFi and money markets:\n{defi_achievements}\n\n"
        f"Evidence of participation in web3 communities:\n{community_achievements}\n\n"
        f"Evidence of engagements within the web3 ecosystem:\n{vibe_achievements}"
    )


def parse_nft(response):
    number_of_nfts = response["story"]["numberOfNftsOwned"]

    nft_achievements = generate_info_list(response["story"]["nftAchievements"])

    return (
        f"They own {number_of_nfts} NFTs including:\n{nft_achievements}\n\n"
    )


def parse_score(response):
    score = response["score"]
    return (
        f"They have a Gitcoin Passport score of: {score}"
    )


def parse_passport(response):

    items = []
    for item in response["items"]:
        if item['metadata']:
            name = item['metadata']['name']
            desc = item['metadata']['description']
            pprint(name)
            items.append(f"* **{name}**: {desc}")
            # items.append(f"* **{name}**")

    full_message = f"They own:\n" + "\n".join(items)

    # Splitting the message into 2000 characters chunks
    return [full_message[i:i+1800] for i in range(0, len(full_message), 1800)]

    # return (
    #     f"They own:\n"
    #     + "\n".join(items))


class ChatBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_responses = {}  # A dictionary to cache API responses

    async def on_ready(self):
        print("Bot 2 is ready!")

    async def on_message(self, message):
        if message.author == self.user:  # Ignore bot's own messages
            return

        if message.content.startswith("!cache"):
            print("CACHE: \n")
            pprint(self.api_responses)
            if not self.api_responses:
                return None

            addresses = list(self.api_responses.keys())
            # TODO change address to the last key in cache
            await message.channel.send(f'latest wallet address = {addresses[-1]}')
            # await message.channel.send(self.api_responses[])

        if message.content.startswith('!score '):
            address = message.content.split(" ")[1]
            await message.channel.send(f"Fetching gitcoin passport score from {address}. This may take a moment...")

            # Check if the data for this ENS domain is already in cache
            if address in self.api_responses:
                await message.channel.send(f"{address} data in database")
                await message.channel.send(parse_score(self.api_responses[address]))
                return

            GET_PASSPORT_SCORE_URI = f"https://api.scorer.gitcoin.co/registry/v2/score/698/{address}"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(GET_PASSPORT_SCORE_URI, headers=passport_headers) as response:
                        if response.status != 200:
                            await message.channel.send(f"Error {response.status}: Unable to retrieve passport score for the provided address.")
                            return

                       # test
                        # pprint(response)
                        data = await response.json()
                        pprint(data)
                        # Check if the data is not None
                        if data is not None:
                            await message.channel.send(f"Successfully got passport score data!")
                            await message.channel.send(parse_score(data))
                        else:
                            await message.channel.send(f"Error: Passport score is None")

            #     # if data.get('success') and data.get('story'):
            #     #     pprint(data)
            #     #     self.api_responses[ens_domain] = data
            #     #     with open('local_state.pkl', 'wb') as f:
            #     #         pickle.dump(self.api_responses, f)
            #     #     await message.channel.send(parse_api_response(data))
            #     # else:
            #     #     await message.channel.send("Unable to retrieve chain history for the provided ENS domain.")

            except Exception as e:
                await message.channel.send(f"Error fetching data: {str(e)}")

        if message.content.startswith('!passport '):
            address = message.content.split(" ")[1]

            await message.channel.send(f"Fetching gitcoin passport from {address}. This may take a moment...")

            # Check if the data for this ENS domain is already in cache
            # if address in self.api_responses:
            #     await message.channel.send(f"{address} data in database")
            #     passport_data_chunks = parse_passport(data)
            #     for chunk in passport_data_chunks:
            #         await message.channel.send(chunk)
            #     # await message.channel.send(parse_passport(self.api_responses[address]))
            #     return

            # [TODO] replace with calling API
            # with open('sample-passport.json') as f:
            #     sample_passport = json.loads(f.read())

            # self.api_responses[address] = sample_passport
            # pprint(sample_passport)
            # # parse_passport(sample_passport)
            # await message.channel.send(parse_passport(sample_passport))

            GET_PASSPORT_STAMPS_URI = f"https://api.scorer.gitcoin.co/registry/v2/stamps/{address}?limit=1000&include_metadata=true"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(GET_PASSPORT_STAMPS_URI, headers=passport_headers) as response:
                        if response.status != 200:
                            await message.channel.send(f"Error {response.status}: Unable to retrieve passport for the provided address.")
                            return

                        data = await response.json()
                        pprint(data)
                        # Check if the data is not None
                        if data is not None:
                            await message.channel.send(f"Successfully got passport data!")
                            # await message.channel.send(parse_passport(data))
                            passport_data_chunks = parse_passport(data)
                            for chunk in passport_data_chunks:
                                await message.channel.send(chunk)
                        else:
                            await message.channel.send(f"Error: Passport data is None")

            except Exception as e:
                await message.channel.send(f"Error fetching data: {str(e)}")

        if message.content.startswith('tell me about '):
            ens_domain = message.content.split(" ")[3]
            await message.channel.send(f"Fetching on-chain data from {ens_domain}. This may take a moment...")

            # Check if the data for this ENS domain is already in cache
            if ens_domain in self.api_responses:
                await message.channel.send(f"{ens_domain} data in database")
                await message.channel.send(parse_api_response(self.api_responses[ens_domain]))
                return

            await message.channel.send("f{ens_domain} data NOT in database")

            # [TODO] replace with calling API
            # with open('sample-data.json') as f:
            #     sample_data = json.load(f)

            # self.api_responses[ens_domain] = sample_data
            # with open('local_state.pkl', 'wb') as f:
            #     pickle.dump(self.api_responses, f)
            # await message.channel.send(parse_api_response(sample_data))
            # print(self.api_responses)

            api_url = f"https://www.chainstory.xyz/api/story/getStoryFromCache?walletId={ens_domain}"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url) as response:
                        if response.status != 200:
                            await message.channel.send(f"Error {response.status}: Unable to retrieve story for the provided ENS domain.")
                            return

                        data = await response.json()

                if data.get('success') and data.get('story'):
                    pprint(data)
                    self.api_responses[ens_domain] = data
                    with open('local_state.pkl', 'wb') as f:
                        pickle.dump(self.api_responses, f)
                    await message.channel.send(parse_api_response(data))
                else:
                    await message.channel.send("Unable to retrieve chain history for the provided ENS domain.")

            except Exception as e:
                await message.channel.send(f"Error fetching data: {str(e)}")

        if "nft" in message.content.lower():
            if ens_domain in self.api_responses:
                response = self.api_responses[ens_domain]

                nft_achievements = '\n'.join(
                    [f"• {nft['title']}: {nft['description']}" for nft in response["story"]["nftAchievements"]])

                output_msg = (
                    f"They own {response['story']['numberOfNftsOwned']} NFTs including:\n{nft_achievements}"
                )
                await message.channel.send('test')
                await message.channel.send(output_msg)
            else:
                await message.channel.send("Please provide an ENS domain first using 'tell me about' command.")

        if message.content.startswith("explain: "):
            query = message.content[len("explain: "):]
            try:
                response = openai.Completion.create(
                    model="text-davinci-003", prompt=query, temperature=0.6, max_tokens=200)
                await message.channel.send(response.choices[0].text.strip())
            except RateLimitError:
                await message.channel.send("Sorry, I'm getting too many requests right now. Please try again later.")
                # Introducing a delay. Adjust as needed.
                await asyncio.sleep(10)


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    bot = ChatBot(intents=intents)
    bot.run(discord_token)
