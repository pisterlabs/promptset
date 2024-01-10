from fastapi import FastAPI, Request, HTTPException
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

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Load Environment variables
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if not RIOT_API_KEY:
    raise ValueError("RIOT_API_KEY environment variable is not set")
RIOT_API_BASE_URL = "https://na1.api.riotgames.com"  # Regional endpoint
RIOT_API_BASE_REGIONAL_URL = "https://americas.api.riotgames.com"
MASS_REGION = "americas"
REGION = "na1"

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up CORS middleware
origins = [
    "http://localhost:8000",
    "https://lacra-gpt-lol.replit.app/",
    "https://chat.openai.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Retrieve Summoner Name
class SummonerNameRequest(BaseModel):
    summoner_name: str


summoner_name = input("Please enter the summoner name: ")
print("Summoner Name:", summoner_name)
regions = ["na1", "eun1", "euw1", "jp1", "kr", "br1"]
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
mass_regions = ["AMERICAS", "ASIA", "EUROPE", "SEA"]
mass_region = input(
    "Please enter the mass region (AMERICAS, ASIA, EUROPE, SEA). Default is 'AMERICAS': "
)
if mass_region == "":
    mass_region = "AMERICAS"
elif mass_region not in mass_regions:
    while mass_region not in mass_regions:
        mass_region = input(
            "Invalid mass region. Please provide a valid MASS REGION (AMERICAS, ASIA, EUROPE, SEA): "
        )
print("Mass Region:", mass_region)


# User inputs summoner name and API returns PUUID
@app.get("/puuid/{summoner_name}")
async def get_puuid(summoner_name: str, region: str, RIOT_API_KEY: str):
    RIOT_API_URL = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner_name}"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            RIOT_API_URL, headers={"X-Riot-Token": RIOT_API_KEY}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        data = response.json()
        return {"puuid": data["puuid"]}


# Get match ids given a players puuid
@app.get("/match_ids")
async def get_match_ids(puuid):
    url = f"{RIOT_API_BASE_REGIONAL_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers={"X-Riot-Token": RIOT_API_KEY})
        match_ids = []
        if response.status_code == 200:
            match_ids = response.json()
        return match_ids


# From a given match id, get the match details
@app.get("/match_details")
async def get_match_details(match_id) -> Optional[dict]:
    url = f"{RIOT_API_BASE_REGIONAL_URL}/lol/match/v5/matches/{match_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers={"X-Riot-Token": RIOT_API_KEY})
        match_details = {}
        if response.status_code == 200:
            match_details = response.json()
        return match_details


def analyze_performance(
    detailed_match_history: List[Optional[dict]], summoner_id: str
) -> dict:
    wins = []
    total_kills = []
    total_deaths = []
    total_assists = []
    total_wards_placed = []
    total_cs = []
    total_games = len(detailed_match_history)

    for match in detailed_match_history:
        if match:
            for participant in match["info"]["participants"]:
                if participant["puuid"] == summoner_id:
                    total_kills += participant["kills"]
                    total_deaths += participant["deaths"]
                    total_assists.append(participant["assists"])
                    total_wards_placed.append(participant["wardsPlaced"])
                    total_cs.append(participant["totalMinionsKilled"])
                    if participant["win"]:
                        wins.append(1)
                    break

    if total_games == 0:
        return {"message": "No recent games found to analyze."}

    win_rate = (sum(wins) / total_games) * 100
    avg_kills = sum(total_kills) / total_games
    avg_deaths = sum(total_deaths) / total_games
    avg_assists = sum(total_assists) / total_games
    avg_wards_placed = sum(total_wards_placed) / total_games
    avg_cs = sum(total_cs) / total_games

    analysis_summary = (
        f"Out of the last {total_games} games, you won {wins} ({win_rate:.2f}% win rate). "
        f"Your average KDA was {avg_kills:.2f}/{avg_deaths:.2f}/{avg_assists:.2f}."
    )

    return {
        "analysis_summary": analysis_summary,
        "win_rate": win_rate,
        "avg_kills": avg_kills,
        "avg_deaths": avg_deaths,
        "avg_assists": avg_assists,
        "avg_wards_placed": avg_wards_placed,
        "avg_cs": avg_cs,
    }


# Update the get_detailed_match_history function definition
async def get_detailed_match_history(
    client: httpx.AsyncClient, match_ids: List[str]
) -> List[Optional[dict]]:
    tasks = [
        get_match_details(client, match_id) for match_id in match_ids
    ]  # Pass the client to get_match_details
    return await asyncio.gather(*tasks)


# Analyze using ChatGPT the performance of the user based on their recent match history
@app.post("/v1/analyze/")
async def analyze_route(summoner: SummonerNameRequest):
    async with httpx.AsyncClient() as client:
        summoner_id = await get_puuid(summoner.summoner_name)
        if summoner_id is not None:
            match_ids = await get_match_ids(summoner_id)
            detailed_match_history = await get_detailed_match_history(
                client, match_ids
            )  # Pass the client along with the match_ids
            analysis = analyze_performance(detailed_match_history, summoner_id)
            # Generate the prompt using the results from the analysis
            win_rate = analysis["win_rate"]
            avg_kills = analysis["avg_kills"]
            avg_deaths = analysis["avg_deaths"]
            avg_assists = analysis["avg_assists"]
            avg_kda = (
                (avg_kills + avg_assists) / avg_deaths
                if avg_deaths > 0
                else float("inf")
            )

            # Prompts
            prompt = (
                f"The player has a win rate of {win_rate:.2f}% with an average KDA of {avg_kda: .2f}."
                f"My win rate is {analysis['win_rate']}, and my average KDA is {analysis['avg_kills']}/{analysis['avg_deaths']}/{analysis['avg_assists']}."
                f"They often play the champion {analysis['most_played_champion']}."
                "What advice would you give to this player to improve their gameplay?"
            )
            # OpenAI API routes
            client = OpenAI()
            # Create a chat completion with the OpenAI API
            response = client.chat.completions.create(
                model="gpt-4",
                # Define the conversation messages
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI esports coach and data analyst.",
                    },
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": "Based on your recent games, here are some insights.",
                    },
                ],
            )
            # Extract the AI's message from the response
            advice = response.choices[0].message.content
            # Add the advice to the analysis dictionary
            analysis["advice"] = advice

            return advice


@app.get("/")
def home():
    return {"message": "Hello, world."}


@app.get("/logo.png")
async def plugin_logo():
    return FileResponse("logo.png", media_type="image/png")


@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    with open("ai-plugin.json", "r") as f:
        json_content = f.read()
    return Response(content=json_content, media_type="application/json")


@app.get("/openapi.yaml")
async def openapi_spec(request: Request):
    host = request.client.host if request.client else "localhost"
    with open("openapi.yaml", "r") as f:
        yaml_content = f.read()
    yaml_content = yaml_content.replace("PLUGIN_HOSTNAME", f"https://{host}")
    return Response(content=yaml_content, media_type="application/yaml")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Create a logger object
logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")

    # Safer error logging
    if response.status_code != 200:
        content = getattr(response, "body", None)
        if content:
            logger.error(f"Error: {content.decode('utf-8')}")
        else:
            # If 'body' is not available, log without content
            logger.error(f"Error: No content available in response")

    return response


# FastAPI routes and tests
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
