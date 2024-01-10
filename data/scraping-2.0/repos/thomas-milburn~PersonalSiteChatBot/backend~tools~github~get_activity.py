import json
from typing import Optional, List, Type

import aiohttp
import requests
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel

import config


class Empty(BaseModel):
    class Config:
        extra = "forbid"


class GithubGetUserEvents(BaseTool):
    name = "github_get_user_events"
    description = """Returns a list of recent events that Thomas has taken on GitHub"""
    args_schema: Type[Empty] = Empty

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        github_username = config.config["PERSONAL_SITE_MY_GITHUB_USERNAME"]
        response = requests.get(f"https://api.github.com/users/{github_username}/events/public")

        if response.status_code == 200:
            return minify_data(response.json())

        return json.dumps([{"error": "Failed to get user events"}])

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        github_username = config.config["PERSONAL_SITE_MY_GITHUB_USERNAME"]
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.github.com/users/{github_username}/events/public") as response:
                if response.status == 200:
                    return minify_data(await response.json())

            return json.dumps([{"error": "Failed to get user events"}])


def minify_data(data: List[dict]) -> str:
    response_data = []

    for event in data:
        event_type = event["type"]
        minified = None

        if event_type == "PushEvent":
            minified = minify_push_event(event)

        if event_type == "PullRequestEvent":
            minified = minify_pull_request_event(event)

        if event_type == "ForkEvent":
            minified = minify_fork_event(event)

        if minified:
            response_data.append(minified)

    return json.dumps(response_data[:10])[:10000]


def minify_push_event(push_event_data) -> dict:
    new_commits = []

    for commit in push_event_data["payload"]["commits"]:
        new_commits.append(commit["message"])

    return {
        "type": "PushEvent",
        "repo": push_event_data["repo"],
        "commits": new_commits,
        "created_at": push_event_data["created_at"],
        "url": f"https://github.com/{push_event_data['repo']['name']}/commit/{push_event_data['payload']['head']}"
    }


def minify_pull_request_event(pull_request_event_data) -> dict:
    return {
        "type": "PullRequestEvent",
        "repo": pull_request_event_data["repo"],
        "action": pull_request_event_data["payload"]["action"],
        "pull_request": {
            "title": pull_request_event_data["payload"]["pull_request"]["title"],
            "description": pull_request_event_data["payload"]["pull_request"]["body"]
        },
        "created_at": pull_request_event_data["created_at"]
    }


def minify_fork_event(fork_event_data) -> dict:
    return {
        "type": "ForkEvent",
        "repo": fork_event_data["repo"],
        "created_at": fork_event_data["created_at"]
    }


if __name__ == "__main__":
    # print(GithubGetUserEvents()._run())
    import asyncio

    loop = asyncio.get_event_loop()
    print(loop.run_until_complete(GithubGetUserEvents()._arun()))
