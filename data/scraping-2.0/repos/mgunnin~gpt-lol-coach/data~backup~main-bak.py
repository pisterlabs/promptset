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
import os
import uvicorn

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Load Environment variables
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if RIOT_API_KEY is None:
    raise ValueError("RIOT_API_KEY environment variable is not set")
RIOT_API_BASE_URL = "https://na1.api.riotgames.com"  # Regional endpoint
RIOT_API_BASE_REGIONAL_URL = "https://americas.api.riotgames.com"

# Initialize FastAPI app
app = FastAPI()

# Configure logging
# logging.basicConfig(level=logging.INFO)

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


@app.get("/summoner_name")
async def get_summoner_id(summoner_name: str) -> Optional[str]:
    async with httpx.AsyncClient() as client:
        url = f"{RIOT_API_BASE_URL}/lol/summoner/v4/summoners/by-name/{summoner_name}"
        response = await client.get(url, headers={"X-Riot-Token": RIOT_API_KEY})
        if response.status_code == 200:
            data = response.json()
        return data["puuid"]  # Return the PUUID instead of the summonerId


@app.get("/matches/{match_id}")
async def get_match_details(match_id: str) -> Optional[dict]:
    async with httpx.AsyncClient() as client:
        url = f"{RIOT_API_BASE_REGIONAL_URL}/lol/match/v5/matches/{match_id}"
        response = await client.get(url, headers={"X-Riot-Token": RIOT_API_KEY})
        if response.status_code == 200:
            return response.json()
        return None


@app.get("/matches/{summoner_id}")
async def get_detailed_match_history(summoner_id: str) -> List[Optional[dict]]:
    async with httpx.AsyncClient() as client:
        try:
            url = f"{RIOT_API_BASE_REGIONAL_URL}/lol/match/v5/matches/by-puuid/{summoner_id}/ids"
            response = await client.get(url, headers={"X-Riot-Token": RIOT_API_KEY})
            if response.status_code == 200:
                match_ids = response.json()
                return await asyncio.gather(
                    *[get_match_details(client, match_id) for match_id in match_ids]
                )
            else:
                return []  # return an empty list if the call doesn't succeed
        except Exception as e:
            print(e)
            return []  # return an empty list if the call doesn't succeed


def analyze_performance(
    detailed_match_history: List[Optional[dict]], summoner_id: str
) -> dict:
    wins = 0
    total_kills = 0
    total_deaths = 0
    total_assists = 0
    total_games = len(detailed_match_history)

    for match in detailed_match_history:
        if match:  # Ensure match details are present
            for participant in match["info"]["participants"]:
                if (
                    participant["puuid"] == summoner_id
                ):  # Update with the correct puuid check
                    total_kills += participant["kills"]
                    total_deaths += participant["deaths"]
                    total_assists += participant["assists"]
                    if participant["win"]:
                        wins += 1
                    break

    if total_games == 0:
        return {"message": "No recent games found to analyze."}

    win_rate = (wins / total_games) * 100
    avg_kills = total_kills / total_games
    avg_deaths = total_deaths / total_games
    avg_assists = total_assists / total_games

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
    }


# FastAPI routes
@app.post("/v1/analyze/")
async def analyze_route(summoner: SummonerNameRequest):
    async with httpx.AsyncClient() as client:
        summoner_id = await get_summoner_id(client, summoner.summoner_name)
        if summoner_id is not None:
            detailed_match_history = await get_detailed_match_history(summoner_id)
        analysis = analyze_performance(detailed_match_history, summoner_id)

        # Generate the prompt using the results from the analysis
        win_rate = analysis["win_rate"]
        avg_kills = analysis["avg_kills"]
        avg_deaths = analysis["avg_deaths"]
        avg_assists = analysis["avg_assists"]
        avg_kda = (
            (avg_kills + avg_assists) / avg_deaths if avg_deaths > 0 else float("inf")
        )

        # Prompts
        prompt = (
            f"The player has a win rate of {win_rate:.2f}% with an average KDA of {avg_kda: .2f}."
            f"My win rate is {analysis['win_rate']}, and my average KDA is {analysis['avg_kills']}/{analysis['avg_deaths']}/{analysis['avg_assists']}."
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
        advice = response["choices"][0]["message"]["content"]

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
    host = request.client.host
    with open("openapi.yaml", "r") as f:
        yaml_content = f.read()
    yaml_content = yaml_content.replace("PLUGIN_HOSTNAME", f"https://{host}")
    return Response(content=yaml_content, media_type="application/yaml")



# Run the async function using the event loop
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
