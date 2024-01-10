import base64
from typing import Optional, Type

import aiohttp
import requests
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class GetReadMeArgs(BaseModel):
    repo_owner: str = Field(description="The repo owner")
    repo_name: str = Field(description="The repo name")


class GithubGetRepoReadMe(BaseTool):
    name = "github_repo_read_me_contents"
    description = """Returns the contents of the README.md file for a given repo"""
    args_schema: Type[GetReadMeArgs] = GetReadMeArgs

    def _run(
        self,
        repo_owner: str,
        repo_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        response = requests.get(f"https://api.github.com/repos/{repo_owner}/{repo_name}/readme")

        if response.status_code != 200:
            return "Failed to get README.md file"

        # Split content into lines and decode from base64
        data = response.json()
        content = data["content"]
        content = content.split("\n")
        if data["encoding"] != "base64":
            return "Failed to get README.md file"

        return "\n".join([base64.b64decode(line).decode("utf-8") for line in content])

    async def _arun(
        self,
        repo_owner: str,
        repo_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.github.com/repos/{repo_owner}/{repo_name}/readme") as response:
                if response.status != 200:
                    return "Failed to get README.md file"

                data = await response.json()
                content = data["content"]
                content = content.replace("\n", "")
                if data["encoding"] != "base64":
                    return "Failed to get README.md file"

                return base64.b64decode(content).decode("utf-8")


# Run _arun (async)
import asyncio
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    print(loop.run_until_complete(GithubGetRepoReadMe()._arun("thomas-milburn", "langchain")))
