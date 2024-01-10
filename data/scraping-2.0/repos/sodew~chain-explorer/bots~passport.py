import discord
import json
import requests
import aiohttp
import re
import datetime
import openai
from pprint import pprint
from web3 import Web3
import pickle

from openai.error import RateLimitError
import asyncio

import os
from dotenv import load_dotenv

load_dotenv()

# Load tokens from .env file
discord_token = os.getenv('DISCORD_TOKEN')
openai_token = os.getenv('OPENAI_API_KEY')
passport_token = os.getenv('PASSPORT_API_KEY')
scorer_token = os.getenv('PASSPORT_SCORER')

# Setup Alechmy API
alchemy_url = "https://eth-mainnet.g.alchemy.com/v2/kyvyHty9Uu7gLaas166z5rFBPsxQDDqT"
w3 = Web3(Web3.HTTPProvider(alchemy_url))

# Setup Passport API
if passport_token:
    passport_headers = {
        'Content-Type': 'application/json',
        'X-API-Key': passport_token
    }


# Setup Passport context
with open('../sample_data/stamp-metadata.json', 'r') as f:
    stamp_metadata = json.load(f)


def parse_passport(response):
    items = []
    for item in response["items"]:
        if item['metadata']:
            name = item['metadata']['name']
            desc = item['metadata']['description']
            pprint(name)
            items.append(f"* **{name}**: {desc}")

    full_message = f"## **They own the following stamps:**\n" + \
        "\n".join(items)

    # Splitting the message into 2000 characters chunks
    return [full_message[i:i+1800] for i in range(0, len(full_message), 1800)]

    # return (
    #     f"They have the following Gitcoin Passport stamps:\n"
    #     + "\n".join(items))


def parse_passport_detailed(response):
    items = []
    for item in response["items"]:
        if item['metadata']:
            name = item['metadata']['name']
            desc = item['metadata']['description']
            group = item['metadata']['group']
            pprint(name)
            items.append(
                f"### **{name}** \nThis stamp is part of the {group}. The holder of this stamp {desc}.")
            # items.append(f"* **{name}**: {desc}")

    full_message = f"## **They own the following stamps:**\n" + \
        "\n".join(items)

    # Splitting the message into 2000 characters chunks
    return [full_message[i:i+1800] for i in range(0, len(full_message), 1800)]


def parse_passport_groups(response, target_groups, category):
    items = []
    for item in response["items"]:
        if item['metadata']:
            name = item['metadata']['name']
            desc = item['metadata']['description']
            task = item['metadata']['platform']['description']
            group = item['metadata']['group']
            # Only process items belonging to any of the target groups
            if group in target_groups:
                # name = item['metadata']['name']
                # desc = item['metadata']['description']
                pprint(name)
                items.append(
                    f"### **{name}** \nThe holder of this stamp {desc}. To get this stamp, they had to {task}")

    full_message = f"## **They own the following stamps in the {category} category:**\n" + \
        "\n".join(items)

    # Splitting the message into 2000 characters chunks
    return [full_message[i:i+1800] for i in range(0, len(full_message), 1800)]

# [TODO] replace this with better data that includes weights


def analyze_stamps(user_stamps, stamp_metadata):
    # Initialize counters for different categories of stamps
    gitcoin_involvement = 0
    human_likelihood = 0

    # Iterate over the user's stamps
    for stamp in user_stamps:
        # Look up the stamp in the metadata
        stamp_info = stamp_metadata.get(stamp['id'])
        if not stamp_info:
            continue

        # Update counters based on the stamp's group
        if stamp_info['group'] == 'Gitcoin':
            gitcoin_involvement += 1
        elif stamp_info['group'] == 'Human Verification':
            human_likelihood += 1

    # Return a summary of the user's involvement
    return {
        'gitcoin_involvement': gitcoin_involvement,
        'human_likelihood': human_likelihood,
    }


# Setup OpenAI API
openai.api_key = openai_token


def classify_message(message):
    prompt = f"""
    Analyze the following message and assign a probability score (from 0.0 to 1.0) indicating how likely it is that the message relates to each of the following categories:
    
    Message: "{message}"

    Probability Scores:
    Humanity: [probability score]
    Socialness: [probability score]
    Gitcoin involvement: [probability score]
    Technical expertise: [probability score]
    Something else: [probability score]

    Only provide the scores, without additional explanation.
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=60
    )
    # Debug: Print raw response text
    print("Raw Response:", response.choices[0].text)

    # Extract and process the response
    lines = response.choices[0].text.strip().split("\n")
    categories = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            category, score = parts[0].strip(), parts[1].strip()
            try:
                categories[category] = float(score)
            except ValueError:
                pass  # Ignore lines that do not have a valid numeric value

    if categories:
        max_category = max(categories, key=categories.get)
        return max_category
    else:
        return "Unable to classify"


def parse_api_response(response):
    wallet_address = response["story"]["walletId"]
    ens_domain = response["story"]["ensName"]
    creation_date = datetime.datetime.fromtimestamp(
        response['story']['walletDOBTimestamp']).strftime("%B %d, %Y")
    latest_transaction_date = datetime.datetime.fromtimestamp(
        response['story']['latestTransactionDateTimestamp']).strftime("%B %d, %Y")

    passport = ""
    print(response["passport"])
    if response["passport"]:
        passport_score = response["passport"]["score"]
        passport_timestamp = datetime.datetime.fromisoformat(
            response["passport"]["last_score_timestamp"]).strftime("%B %d, %Y")
        passport = f"They have a Gitcoin Passport score of {passport_score} as of {passport_timestamp}"
    else:
        print("no passport response")

    number_of_nfts = response["story"]["numberOfNftsOwned"]

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

    nft_achievements = generate_description_list(
        response["story"]["nftAchievements"])
    defi_achievements = generate_description_list(
        response["story"]["deFiAchievements"])
    community_achievements = generate_description_list(
        response["story"]["communityAchievements"])
    vibe_achievements = generate_info_list(
        response["story"]["vibeAchievements"])

    return (
        f"The wallet {wallet_address} belongs to **{ens_domain}**. This wallet was "
        f"created on {creation_date} and their latest transaction was on {latest_transaction_date}.\n\n"
        f"They own {number_of_nfts} NFTs including:\n{nft_achievements}\n\n"
        f"## **Evidence of their participation in DeFi and money markets**:\n{defi_achievements}\n\n"
        f"## **Evidence of participation in web3 communities**:\n{community_achievements}\n\n"
        f"## **Evidence of engagements within the web3 ecosystem**:\n{vibe_achievements}\n\n"
        f"### **{passport}**"
    )


class ChatBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_responses = {}  # A dictionary to cache API responses

    async def on_ready(self):
        print("Bot is ready!")

    async def on_message(self, message):
        if message.author == self.user:  # Ignore bot's own messages
            return

        if message.content.startswith("!cache"):
            print("CACHE: \n")
            pprint(self.api_responses)
            # If the cache is empty, return None
            if not self.api_responses:
                await message.channel.send("Please input a wallet addresss to analyze. Then ask again.")
                return None

            addresses = list(self.api_responses.keys())
            await message.channel.send(f"addresses: {addresses}")
            await message.channel.send(f'latest wallet address = {addresses[-1]}')

        if '0x' in message.content or '.eth' in message.content:
            eth_address_pattern = re.compile(r"0x[a-fA-F0-9]{40}")
            eth_domain_pattern = re.compile(r"\b\w+\.eth\b")
            address = ''

            if '0x' in message.content:
                address_match = re.search(eth_address_pattern, message.content)
                if address_match:
                    address = address_match.group()
                    # await message.channel.send(f"Address: {address}")

            elif '.eth' in message.content:
                domain_match = re.search(eth_domain_pattern, message.content)
                if domain_match:
                    address = domain_match.group()
                    # await message.channel.send(f"Address: {address}")

            await message.channel.send(f"Fetching on-chain data from {address}. This may take a moment...")

            # Check if the data for this address is already in cache
            if address in self.api_responses:
                await message.channel.send(f"{address} in cache")
                await message.channel.send(parse_api_response(self.api_responses[address]))
                return

            CHAINSTORY_URI = f"https://www.chainstory.xyz/api/story/getStoryFromCache?walletId={address}"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(CHAINSTORY_URI) as response:
                        if response.status != 200:
                            await message.channel.send(f"Error {response.status}: Unable to retrieve chainstory for the provided ethereum address.")
                            return

                        data = await response.json()

                if data.get('success') and data.get('story'):
                    pprint(data)
                    self.api_responses[address] = data

                    # Retrieve more data from Gitcoin Passport
                    passport_address = data["story"]["walletId"]
                    print(passport_address)
                    GET_PASSPORT_SCORE_URI = f"https://api.scorer.gitcoin.co/registry/v2/score/{scorer_token}/{passport_address}"

                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(GET_PASSPORT_SCORE_URI, headers=passport_headers) as passport_response:
                                if passport_response.status == 200:
                                    passport_data = await passport_response.json()
                                    # await message.channel.send(f"User {address} has a Gitcoin Passport")

                                    # Adding the passport score data into the nested dictionary
                                    self.api_responses[address]['passport'] = passport_data
                                else:
                                    self.api_responses[address]['passport'] = {
                                    }
                                    print(
                                        f"Error {passport_response.status}: Unable to retrieve passport score for the address.")
                                    await message.channel.send(f"User {address} does not have a Gitcoin Passport")

                    except Exception as e:
                        await message.channel.send(f"Error fetching passport score: {str(e)}")

                    # with open('local_state.pkl', 'wb') as f:
                    #     pickle.dump(self.api_responses, f)
                    await message.channel.send(parse_api_response(self.api_responses[address]))
                else:
                    await message.channel.send("Unable to retrieve chain information for the provided wallet address.")

            except Exception as e:
                error_msg = f"Error fetching data: {str(e)}"
                print(error_msg)
                await message.channel.send(error_msg)

        # passport stamps prompt
        if message.content.startswith("!passport stamps"):
            if not self.api_responses:
                await message.channel.send("Please input a wallet addresss to analyze. Then ask again.")
                return None

            cached_addresses = list(self.api_responses.keys())
            latest_address = cached_addresses[-1]
            address = self.api_responses[latest_address]['story']['walletId']
            await message.channel.send(address)
            pprint(self.api_responses[latest_address])

            if 'passportStamps' in self.api_responses[latest_address] and self.api_responses[latest_address]['passportStamps']:
                await message.channel.send('passport stamps in cache')
                data = self.api_responses[latest_address]['passportStamps']
                # Check if the data is not None
                if data is not None:
                    await message.channel.send(f"Successfully got passport data!")

                    passport_data_chunks = parse_passport_detailed(data)
                    for chunk in passport_data_chunks:
                        await message.channel.send(chunk)

                    self.api_responses[latest_address]['passportStamps'] = data
                else:
                    await message.channel.send(f"Error: Passport data is None")

            else:
                GET_PASSPORT_STAMPS_URI = f"https://api.scorer.gitcoin.co/registry/stamps/{address}?limit=1000&include_metadata=true"

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

                                passport_data_chunks = parse_passport_detailed(
                                    data)
                                for chunk in passport_data_chunks:
                                    await message.channel.send(chunk)

                                self.api_responses[latest_address]['passportStamps'] = data
                            else:
                                await message.channel.send(f"Error: Passport data is None")

                except Exception as e:
                    await message.channel.send(f"Error fetching data: {str(e)}")

        # alchemy prompt
        if message.content.startswith("!alchemy"):
            if not self.api_responses:
                await message.channel.send("Please input a wallet addresss to analyze. Then ask again.")
                return None

            cached_addresses = list(self.api_responses.keys())
            latest_address = cached_addresses[-1]
            address = self.api_responses[latest_address]['story']['walletId']
            # print("is connected? " + w3.is_connected())
            await message.channel.send(f"is connected? " + str(w3.is_connected()))
            await message.channel.send(f"ETH balance: " + str(w3.eth.get_balance(address)))

        if message.content.startswith("!explain "):
            query = message.content[len("!explain "):]
            prompt = f"As an expert in Gitcoin Passport, your role is to provide detailed information and insights about the Gitcoin Passport application, its purpose, and its use, particularly focusing on the various stamps it offers. Gitcoin Passport is a vital identity verification application and Sybil resistance protocol, designed to enhance the security and integrity of digital communities and projects. Your responses should be informed, precise, and helpful to users seeking to understand how Gitcoin Passport works, especially in regards to its verifiable credentials or Stamps. Explain in one clear and concise sentence  {query} "
            try:
                response = openai.Completion.create(
                    model="text-davinci-003", prompt=prompt, temperature=0.6, max_tokens=200)
                await message.channel.send(response.choices[0].text.strip())
            except RateLimitError:
                await message.channel.send("Sorry, I'm getting too many requests right now. Please try again later.")
                # Introducing a delay. Adjust as needed.
                await asyncio.sleep(10)

        if message.content.startswith("!test "):
            if not self.api_responses:
                await message.channel.send("Please input a wallet addresss to analyze. Then ask again.")
                return None

            cached_addresses = list(self.api_responses.keys())
            latest_address = cached_addresses[-1]
            await message.channel.send(latest_address)

            passportStamps = {}
            if 'passportStamps' in self.api_responses[latest_address]:
                passportStamps = self.api_responses[latest_address]['passportStamps']

            passport = {}
            if 'passport' in self.api_responses[latest_address]:
                passport = self.api_responses[latest_address]['passport']

            
            if passport is not None:
                print(passport['score'])
                await message.channel.send(passport['score'])
            else:
                await message.channe.send('passport is none')

            if passportStamps is not None:
                await message.channel.send(f"Successfully got passport data!")
                target_groups = ["Account Name",  "Account Creation", "Government ID", "CAPTCHA Pass", "Uniqueness Pass", "Liveness Pass"]
                category = 'Humanity'
                passport_data_chunks = parse_passport_groups(
                    passportStamps, target_groups, category)
                for chunk in passport_data_chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(f"Error: Passport stamp data is None")
            # if not self.api_responses:
            #     await message.channel.send("Please input a wallet addresss to analyze. Then ask again.")
            #     return None

            # cached_addresses = list(self.api_responses.keys())
            # latest_address = cached_addresses[-1]
            # await message.channel.send(latest_address)

            # if 'passportStamps' in self.api_responses[latest_address] and self.api_responses[latest_address]['passportStamps']:
            #     await message.channel.send('passport stamps in cache')
            #     data = self.api_responses[latest_address]['passportStamps']
            #     # Check if the data is not None
            #     if data is not None:
            #         await message.channel.send(f"Successfully got passport data!")
            #         target_groups = ["Account Name", "NFT Holder"]
            #         category = 'Humanity'
            #         passport_data_chunks = parse_passport_groups(
            #             data, target_groups, category)
            #         for chunk in passport_data_chunks:
            #             await message.channel.send(chunk)

            #         self.api_responses[latest_address]['passportStamps'] = data
            #     else:
            #         await message.channel.send(f"Error: Passport stamp data is None")

            # else:
            #     await message.channel.send('passport stamps NOT in cache')

        # [TODO] add more here for other asks
        if message.content.startswith("!ask "):
            query = message.content[len("!ask "):]
            category = classify_message(query)
            await message.channel.send(f"This question was classified as: " + category)

            if not self.api_responses:
                await message.channel.send("Please input a wallet addresss to analyze. Then ask again.")
                return None

            cached_addresses = list(self.api_responses.keys())
            latest_address = cached_addresses[-1]
            await message.channel.send(latest_address)

            passportStamps = {}
            if 'passportStamps' in self.api_responses[latest_address]:
                passportStamps = self.api_responses[latest_address]['passportStamps']

            passport = {}
            if 'passport' in self.api_responses[latest_address]:
                passport = self.api_responses[latest_address]['passport']

            if category == "Humanity":
                if passport is not None:
                    print(passport['score'])
                    passport_score = passport['score']
                    await message.channel.send(f"The user " + latest_address + " has a passport score of " + passport_score)
                    # [TODO] add in stamps into chat gpt to add extra knowledge. 
                    prompt = f"As an expert in Gitcoin Passport, your role is to provide detailed information and insights about the humanity of the user. Gitcoin Passport is a vital identity verification application and Sybil resistance protocol, designed to enhance the security and integrity of digital communities and projects. Your responses should be informed, precise, and helpful to users in determing humanity. Explain how likely the user is to be human based on a passport score of {passport_score} and the ownership of the following stamps."
                    try:
                        response = openai.Completion.create(
                            model="text-davinci-003", prompt=prompt, temperature=0.6, max_tokens=200)
                        await message.channel.send(response.choices[0].text.strip())
                    except RateLimitError:
                        await message.channel.send("Sorry, I'm getting too many requests right now. Please try again later.")
                        # Introducing a delay. Adjust as needed.
                        await asyncio.sleep(10)


                if passportStamps is not None:
                    # [TODO] edit target_groups to those that represent humanity
                    target_groups = ["Account Name",  "Account Creation", "Government ID", "CAPTCHA Pass", "Uniqueness Pass", "Liveness Pass"]
                    category = 'humanity'
                    passport_data_chunks = parse_passport_groups(
                        passportStamps, target_groups, category)
                    for chunk in passport_data_chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(f"Error: Passport stamp data is None")

                return
            elif category == "Gitcoin involvement":
                if passportStamps is not None:
                    await message.channel.send(f"Successfully got passport data!")
                    target_groups = ["Self GTC Staking", "Contributed to...", "Contributed ($)..."]
                    category = 'gitcoin involvement'
                    passport_data_chunks = parse_passport_groups(
                        passportStamps, target_groups, category)
                    for chunk in passport_data_chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(f"Error: Passport stamp data is None")
                return
            elif category == "Technical expertise":
                if passportStamps is not None:
                    await message.channel.send(f"Successfully got passport data!")
                    target_groups = ["Contribution Activity", "Possessions", "Transactions", "NFT Holder"]
                    category = 'technical expertise'
                    passport_data_chunks = parse_passport_groups(
                        passportStamps, target_groups, category)
                    for chunk in passport_data_chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(f"Error: Passport stamp data is None")
                return
            elif category == "socialness":
                if passportStamps is not None:
                    await message.channel.send(f"Successfully got passport data!")
                    target_groups = ["Snapshot Voter", "Snapshot Proposal Creator", "Lens Handle", "Guild Member"]
                    category = 'Socialness'
                    passport_data_chunks = parse_passport_groups(
                        passportStamps, target_groups, category)
                    for chunk in passport_data_chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(f"Error: Passport stamp data is None")
                return
            else:
                await message.channel.send(f"Unable to classify your question. Please ask about their humanity, technical expertise, or involvement in gitcoin.")
                return


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    bot = ChatBot(intents=intents)
    bot.run(discord_token)
