# Importing necessary libraries
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from riotwatcher import LolWatcher, ApiError
import httpx
import asyncio
import logging
import os
import requests
import pandas as pd

# Creating a new FastAPI application
app = FastAPI()

# Constants
regions = ["na1", "eun1", "euw1", "jp1", "kr", "br1"]
mass_regions = ["americas", "asia", "europe", "sea"]
no_games = 25
queue_id = 420
RIOT_API_ROUTES = {
    "summoner": "/lol/summoner/v4/summoners/by-name/{summonerName}",
    "match_by_puuid": "/lol/match/v5/matches/by-puuid/{puuid}/ids",
    "match_by_id": "/lol/match/v5/matches/{matchId}",
    "match_timeline": "/lol/match/v5/matches/{matchId}/timeline"
}
QUEUE_ID_ROUTES = {
    "draft_pick": 400,
    "ranked_solo": 420,
    "blind_pick": 430,
    "ranked_flex": 440,
    "aram": 450
}

# Loading Environment variables
load_dotenv()
# Getting the Riot API Key from environment variables
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
# If the Riot API Key is not set, raise an error
if not RIOT_API_KEY:
    raise ValueError("RIOT_API_KEY environment variable is not set")

# Asking the user for their summoner name, region, and mass region
summoner_name = input("Please enter the summoner name: ")
print("Summoner Name:", summoner_name)
region = input(
    "Please enter the region (na1, eun1, euw1, jp1, kr, br1). Default is 'na1': "
)
if region == "":
    region = "na1"
elif region not in regions:
    while region not in regions:
        region = input(
            "Invalid region. Please enter the region (na1, eun1, euw1, jp1, kr, br1): "
        )
print("Region:", region)
mass_region = input(
    "Please enter the mass region (americas, asia, europe, sea). Default is 'americas': "
)
if mass_region == "":
    mass_region = "americas"
elif mass_region not in mass_regions:
    while mass_region not in mass_regions:
        mass_region = input(
            "Invalid mass region. Please provide a valid MASS REGION (americas, asia, europe, sea): "
        )
print("Mass Region:", mass_region)


# This function takes a summoner name, region, and Riot API key and returns the player's puuid.
# The puuid is a unique identifier for a player and is used by the Riot API to identify players.
async def get_puuid(summoner_name, region, RIOT_API_KEY):
    # Making an asynchronous GET request to the Riot API to get the player's puuid
    async with httpx.AsyncClient() as client:
        RIOT_API_URL = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner_name}"
        resp = await client.get(RIOT_API_URL, headers={"X-Riot-Token": RIOT_API_KEY})
    # Parsing the response to get the player's puuid
    resp = requests.get(RIOT_API_URL)
    resp.raise_for_status()
    player_info = resp.json()
    return player_info.get("puuid")


# Function to get the match IDs of a player
# 1. We create a function called `get_match_ids` that takes in 5 arguments: `puuid`, `mass_region`, `no_games`, `queue_id`, and `RIOT_API_KEY`.
# 2. `RIOT_API_URL` is a string that contains the Riot API URL to get the match IDs. We use f-strings to format the variables into the string.
# 3. We send a GET request to the Riot API URL and assign the response to a variable called `resp`.
# 4. We then call the `raise_for_status()` method on `resp` to raise an exception if there is an error in the request.
# 5. Finally, we return the JSON data in the response as a Python dictionary.


def get_match_ids(puuid, mass_region, no_games, queue_id, RIOT_API_KEY):
    # Making a GET request to the Riot API to get the match IDs
    RIOT_API_URL = f"https://{mass_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={no_games}&queue={queue_id}&api_key={RIOT_API_KEY}"
    resp = requests.get(RIOT_API_URL)
    resp.raise_for_status()
    # Parsing the response to get the match IDs
    return resp.json()


# Function to get the match data
async def get_match_data(match_id, mass_region, RIOT_API_KEY):
    # Making an asynchronous GET request to the Riot API to get the match data
    async with httpx.AsyncClient() as client:
        RIOT_API_URL = (
            f"https://{mass_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        )
        response = await client.get(
            RIOT_API_URL, headers={"X-Riot-Token": RIOT_API_KEY}
        )
    # Parsing the response to get the match data
    return response.json()


# Function to find the player data in the match data
async def find_player_data(match_data, puuid):
    # Getting the list of participants in the match
    participants = match_data["metadata"]["participants"]
    # Finding the index of the player in the participants list
    player_index = participants.index(puuid)
    # Returning the player's data
    return match_data["info"]["participants"][player_index]


# Function to gather all the data
async def gather_all_data(puuid, match_ids, mass_region, RIOT_API_KEY):
    # Creating a dictionary to store the data
    data = {"champion": [], "kills": [], "deaths": [], "assists": [], "win": []}
    # For each match ID, get the match data and find the player's data
    for match_id in match_ids:
        match_data = await get_match_data(match_id, mass_region, RIOT_API_KEY)
        # If the match data is not found, skip this match ID
        if "status" in match_data and match_data["status"]["status_code"] == 404:
            print(
                f"Match data not found for match ID: {match_id}. Skipping this match ID."
            )
            continue
        # Find the player's data and add it to the dictionary
        player_data = find_player_data(match_data, puuid)
        data["champion"].append(player_data["championName"])
        data["kills"].append(player_data["kills"])
        data["deaths"].append(player_data["deaths"])
        data["assists"].append(player_data["assists"])
        data["win"].append(player_data["win"])
    # Return the data as a pandas DataFrame
    return pd.DataFrame(data)


# Function to run all the functions and gather all the data
async def master_function(
    summoner_name, region, mass_region, no_games, queue_id, RIOT_API_KEY
):
    # Get the player's puuid
    puuid = await get_puuid(summoner_name, region, RIOT_API_KEY)
    # Get the match IDs
    match_ids = get_match_ids(puuid, mass_region, no_games, queue_id, RIOT_API_KEY)
    # Gather all the data
    return await gather_all_data(puuid, match_ids, mass_region, RIOT_API_KEY)


# Running the master function and storing the data in a DataFrame
df = asyncio.run(
    master_function(
        summoner_name, region, mass_region, no_games, queue_id, RIOT_API_KEY
    )
)

# Adding a count column to the DataFrame
df["count"] = 1
# Grouping the DataFrame by champion and calculating the mean of the kills, deaths, assists, and win columns, and the sum of the count column
champ_df = df.groupby("champion").agg(
    {
        "kills": "mean",
        "deaths": "mean",
        "assists": "mean",
        "win": "mean",
        "count": "sum",
    }
)
# Resetting the index of the DataFrame
champ_df.reset_index(inplace=True)
# Filtering the DataFrame to only include champions that have been played at least 2 games
champ_df = champ_df[champ_df["count"] >= 2]
# Calculating the KDA (Kill/Death/Assist ratio) for each champion
champ_df["kda"] = (champ_df["kills"] + champ_df["assists"]) / champ_df["deaths"]
# Sorting the DataFrame by KDA in descending order
champ_df = champ_df.sort_values("kda", ascending=False)

# Getting the row with the highest KDA
best_row = champ_df.iloc[0]
# Getting the row with the lowest KDA
worst_row = champ_df.iloc[-1]
# Printing the champion with the best and worst KDA
print(
    "Your best KDA is on",
    best_row["champion"],
    "with a KDA of",
    best_row["kda"],
    "over",
    best_row["count"],
    "game/s",
)
print(
    "Your worst KDA is on",
    worst_row["champion"],
    "with a KDA of",
    worst_row["kda"],
    "over",
    worst_row["count"],
    "game/s",
)

# Sorting the DataFrame by count in descending order
champ_df = champ_df.sort_values("count", ascending=False)
# Getting the row with the highest count
row = champ_df.iloc[0]
# Calculating the win rate
win_rate = str(round(row["win"] * 100, 1)) + "%"
# Printing the champion with the highest play count and their win rate
print(
    "Your highest played Champion is",
    row["champion"],
    "with",
    row["count"],
    "game/s",
    "and an average Win Rate of",
    win_rate,
)

# Sorting the DataFrame by kills in descending order
highest_kills = df.sort_values("kills", ascending=False)
# Getting the row with the highest kills
row = highest_kills.iloc[0]
# Printing the champion with the highest kill game
print(
    "Your highest kill game was with",
    row["champion"],
    "where you had",
    row["kills"],
    "kills",
)
