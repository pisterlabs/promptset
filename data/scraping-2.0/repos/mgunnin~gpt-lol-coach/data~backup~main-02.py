from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from riotwatcher import LolWatcher, ApiError
from openai import OpenAI

# Load Environment variables
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

if RIOT_API_KEY is None:
    raise ValueError("API keys not found in environment variables.")

# Riot and OpenAI API setup
lol_watcher = LolWatcher(RIOT_API_KEY)

# FastAPI app initialization
app = FastAPI()


# Pydantic models for structured data
class SummonerName(BaseModel):
    name: str


class PlayerStats(BaseModel):
    kda: dict
    creepScore: int
    goldEarned: int
    damageDealt: int
    damageTaken: int
    visionScore: int
    objectiveControl: dict
    championStats: dict
    historicalPerformance: dict
    comparativeAnalysis: dict


# Function to analyze player stats
def analyze_player_stats(player_data):
    return {
        "kda": {"kills": 10, "deaths": 2, "assists": 5},  # Example values
        "creepScore": 150,
        "goldEarned": 12000,
        "damageDealt": 20000,
        "damageTaken": 10000,
        "visionScore": 30,
        "objectiveControl": {"dragonKills": 3, "baronKills": 1},
        "championStats": {
            "championUsed": "Ahri",
            "winRate": 0.75,
            "preferredRoles": ["Mid"],
        },
        "historicalPerformance": {
            "winLossRatio": 0.6,
            "averageScores": {
                "averageKills": 8,
                "averageDeaths": 4,
                "averageAssists": 6,
            },
        },
        "comparativeAnalysis": {
            "skillLevelComparison": "Above Average",
            "proPlayerComparison": "Below Average",
        },
    }


# Endpoint to retrieve player stats and provide coaching
@app.post("/player_coaching")
async def get_player_coaching(summoner: SummonerName, request: Request):
    try:
        # Retrieve player data from the Riot Games API
        player_data = lol_watcher.summoner.by_name("na1", summoner.name)
        analyzed_stats = analyze_player_stats(player_data)

        # Fetch the conversation history for context
        chat_history = await request.json()
        conversation = chat_history.get("conversation", "")

        # Call OpenAI API for chat-based coaching
        client = OpenAI()
        chat_response = client.chat.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a coach for League of Legends."},
                {"role": "user", "content": conversation},
                {
                    "role": "assistant",
                    "content": "Based on your recent games, here are some insights.",
                },
            ]
            + analyzed_stats,  # Incorporating analyzed stats into the conversation
        )

        return JSONResponse(content=chat_response, status_code=200)
    except ApiError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
