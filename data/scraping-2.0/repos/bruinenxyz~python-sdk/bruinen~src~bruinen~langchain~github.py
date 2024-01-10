# TODO figure out how the generator handles multiple methods (GET, POST, etc.) with the same name

import json
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Callable, List, Optional
from urllib.parse import quote

from ..client import AuthenticatedClient
from ..client.api.accounts import find_all_accounts_for_user
from ..client.api.auth import get_user_auth_token
from ..client.models import Auth, ReturnedAccountDto
from ..client.types import Response

from ..client.api.sources import github_controller_repos
from ..client.models import GithubRepo
from ..client.api.sources import github_controller_profile
from ..client.models import GithubProfile


class GithubAuthenticatorTool(BaseTool):
    name = 'github_authenticator_tool'
    description = """Useful for when a user's Github account is not authenticated.

    Input to the tool should be an empty string.

    The response from the tool will be a URL that you return to the user for them to complete auth.
    The URL will be your final answer.
    """

    client: AuthenticatedClient
    user_id: str
    server: str = 'https://ui.bruinen.co'
    source_policy_id: str = None
    redirect_url: str

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run the tool."""
        response: Response[Auth] = get_user_auth_token.sync_detailed(client=self.client, user_id=self.user_id)
        user_token = response.parsed.access_token

        if self.source_policy_id is not None:
            source = [{ 'name': 'github', 'sourcePolicyId': self.source_policy_id }]
        else:
            source = [{ 'name': 'github' }]
        encoded_source = quote(json.dumps(source))
        
        return self.server + '/connect' + '?userToken=' + quote(user_token) + '&sources=' + encoded_source + '&defaultRedirectUrl=' + quote(self.redirect_url)

    # TODO implement async version
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        return await self._run(query, run_manager)


class GithubGetReposTool(BaseTool):
    name = "github_get_repos_tool"
    description = """Useful for when you need to get a user's Github repos.
    
    Input should be the question that you want to know the answer to.
    
    Output will be the text response from the Github API.
    """

    client: AuthenticatedClient
    user_id: str
    parse_output: Optional[Callable[[List["GithubRepo"]], str]] = None
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool."""
        
        response: Response[List["ReturnedAccountDto"]] = find_all_accounts_for_user.sync_detailed(
            client=self.client, user_id=self.user_id
        )
        if not 200 <= response.status_code < 300:
            return "Error pulling the user's Github account."
        accounts: List["ReturnedAccountDto"] = response.parsed

        account_id = ""
        for account in accounts:
            if account.source == "github":
                account_id = account.id
        if account_id == "":
            return "The user has not connected their Github account; you should try authenticating Github first."
        else:
            response: Response[List["GithubRepo"]] = github_controller_repos.sync_detailed(
                client=self.client,
                account_id=account_id
            )
            if not 200 <= response.status_code < 300:
                return "Error when attempting to get the user's Github repos."

            if self.parse_output is None:
                # Call each response item's to_dict() method and return the result as a JSON string
                return json.dumps(list(map(lambda x: x.to_dict(), response.parsed)))
            else:
                return self.parse_output(response.parsed, query)

    # TODO implement async version
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        return await self._run(query, run_manager)


class GithubGetProfileTool(BaseTool):
    name = "github_get_profile_tool"
    description = """Useful for when you need to get a user's Github profile.
    
    Input should be the question that you want to know the answer to.
    
    Output will be the text response from the Github API.
    """

    client: AuthenticatedClient
    user_id: str
    parse_output: Optional[Callable[[GithubProfile], str]] = None
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool."""
        
        response: Response[List["ReturnedAccountDto"]] = find_all_accounts_for_user.sync_detailed(
            client=self.client, user_id=self.user_id
        )
        if not 200 <= response.status_code < 300:
            return "Error pulling the user's Github account."
        accounts: List["ReturnedAccountDto"] = response.parsed

        account_id = ""
        for account in accounts:
            if account.source == "github":
                account_id = account.id
        if account_id == "":
            return "The user has not connected their Github account; you should try authenticating Github first."
        else:
            response: Response[GithubProfile] = github_controller_profile.sync_detailed(
                client=self.client,
                account_id=account_id
            )
            if not 200 <= response.status_code < 300:
                return "Error when attempting to get the user's Github profile."

            if self.parse_output is None:
                return json.dumps(response.parsed.to_dict())
            else:
                return self.parse_output(response.parsed, query)

    # TODO implement async version
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        return await self._run(query, run_manager)

