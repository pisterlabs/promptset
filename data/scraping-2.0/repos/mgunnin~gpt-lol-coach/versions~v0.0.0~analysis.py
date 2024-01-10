from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from openai import OpenAI
import httpx
import asyncio
import os
from dotenv import load_dotenv


class Analysis:
    def __init__(self):
        pass

    def analyze(self, player_stats):
        # Initialize the analysis result
        analysis_result = {}

        # Check if player_stats is not empty
        if player_stats:
            # Calculate the average kills, deaths, and assists
            total_kills = sum(
                [match["participants"][0]["stats"]["kills"] for match in player_stats]
            )
            total_deaths = sum(
                [match["participants"][0]["stats"]["deaths"] for match in player_stats]
            )
            total_assists = sum(
                [match["participants"][0]["stats"]["assists"] for match in player_stats]
            )
            total_matches = len(player_stats)

            analysis_result["Average Kills"] = total_kills / total_matches
            analysis_result["Average Deaths"] = total_deaths / total_matches
            analysis_result["Average Assists"] = total_assists / total_matches

            # Calculate the win rate
            total_wins = sum(
                [match["participants"][0]["stats"]["win"] for match in player_stats]
            )
            analysis_result["Win Rate"] = (total_wins / total_matches) * 100

            # Calculate the average gold earned
            total_gold_earned = sum(
                [
                    match["participants"][0]["stats"]["goldEarned"]
                    for match in player_stats
                ]
            )
            analysis_result["Average Gold Earned"] = total_gold_earned / total_matches

            # Calculate the average minions killed
            total_minions_killed = sum(
                [
                    match["participants"][0]["stats"]["totalMinionsKilled"]
                    for match in player_stats
                ]
            )
            analysis_result["Average Minions Killed"] = (
                total_minions_killed / total_matches
            )

        return analysis_result


async def analyze_performance(match_history: List[Optional[dict]], puuid: str) -> dict:
    wins = []
    total_kills = []
    total_deaths = []
    total_assists = []
    total_wards_placed = []
    total_cs = []
    total_games = len(match_history)

    for match in match_history:
        if match:
            for participant in match("info", {})("participants", []):
                if participant("puuid") == puuid:
                    total_kills += participant["kills"]
                    total_deaths += participant["deaths"]
                    total_assists += participant["assists"]
                    total_wards_placed += participant["wardsPlaced"]
                    total_cs += participant["totalMinionsKilled"] + participant.get(
                        "neutralMinionsKilled", 0
                    )
                    if participant["win"]:
                        wins += 1
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
        f"Your average KDA was {avg_kills:.2f}/{avg_deaths:.2f}/{avg_assists:.2f}. "
        f"On average, you placed {avg_wards_placed:.2f} wards and killed {avg_cs:.2f} minions per game."
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
