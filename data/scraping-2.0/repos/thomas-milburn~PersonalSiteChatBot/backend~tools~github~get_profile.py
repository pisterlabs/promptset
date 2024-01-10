from typing import Type, Optional

import requests
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class UserSchema(BaseModel):
    username: str = Field(description="should be a github username")


class GithubGetUserProfile(BaseTool):
    name = "github_get_user_profile"
    description = """Returns details about a given GitHub user, including URLs related to their activity, 
    profile information, and public statistics"""
    args_schema: Type[UserSchema] = UserSchema

    def _run(
        self,
        username: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Use the tool."""
        response = requests.get(f"https://api.github.com/users/{username}")
        return response.json()

    async def _arun(
        self,
        username: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
