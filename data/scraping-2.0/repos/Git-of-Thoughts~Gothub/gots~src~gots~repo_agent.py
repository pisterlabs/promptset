from datetime import datetime
from typing import Callable, Optional

## this imports langchain from the langchain package
import langchain

## this imports the Head and Repo classes from the git package
from git import Head, Repo

## this imports the AgentType and initialize_agent functions from the langchain.agents package
from langchain.agents import AgentType, initialize_agent

## this imports the ChatOpenAI class from the langchain.chat_models package
from langchain.chat_models import ChatOpenAI

## this imports the BaseModel and dataclass functions from the pydantic package
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

## this imports the ability_runner_factory and oracle_runner_factory functions from the gots.tools package
from gots.tools.ability_runner import ability_runner_factory
from gots.tools.oracle_runner import oracle_runner_factory

## this imports the GitCallbackHandler class from the gots.callbacks.git_callback_handler package
from .callbacks.git_callback_handler import GitCallbackHandler
from .tools.scoped_file_tools import build_scoped_file_tools

# keep this true if you want to see the outputs
langchain.debug = True


class WriteRepoInp(BaseModel):
    ## This class provides a list of inputs to be used in another directory, gothub, to create a pull request
    repo: Repo
    openai_api_key: str
    extra_prompt: Optional[str]

    class Config:
        ## This class provides a list of configurations to be used in another directory, gothub, to create a pull request
        arbitrary_types_allowed = True


class WriteRepoOut(BaseModel):
    ## This class provides a list of outputs to be used in another directory, gothub, to create a pull request
    new_branches: list[Head]
    answer: str

    class Config:
        ## This class provides a list of configurations to be used in another directory, gothub, to create a pull request
        arbitrary_types_allowed = True


RepoAgent = Callable[
    [WriteRepoInp], WriteRepoOut
]  ## This class provides a list of callable functions to be used in another directory, gothub, to create a pull request


def one_branch_mrkl(inp: WriteRepoInp) -> None:
    ## this function uses the WriteRepoInp class to create a new request and initialize the agent to run the request
    match inp:
        case WriteRepoInp(
            repo=repo,
            openai_api_key=openai_api_key,
            extra_prompt=extra_prompt,
        ):
            pass

    tools = [
        *build_scoped_file_tools(repo.working_dir),
        oracle_runner_factory(repo.working_dir + "/.."),
        ability_runner_factory(repo.working_dir + "/.."),
    ]  ## this creates a list of tools to be used in the agent

    llm = ChatOpenAI(
        temperature=0,
        # model="gpt-3.5-turbo-0613",
        model="gpt-4-0613",
        openai_api_key=openai_api_key,
        callbacks=[GitCallbackHandler(repo)],
    )  ## this creates a ChatOpenAI object to be used in the agent

    mrkl = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        callbacks=[GitCallbackHandler(repo)],
        verbose=False,
    )  ## this initializes the agent

    answer = mrkl.run(extra_prompt)  ## this runs the agent
    return answer  ## this returns the answer
    # mrkl.run(extra_prompt)


def gots_repo_agent(inp: WriteRepoInp) -> WriteRepoOut:
    ## this function uses the WriteRepoInp class to create a new request and uses the one_branch_mrkl function to run the request, returning the answer to the request
    """
    ! Should only modify what's permitted by inp
    """
    match inp:
        case WriteRepoInp(
            repo=repo,
            openai_api_key=openai_api_key,
            extra_prompt=extra_prompt,
        ):
            pass

    time = datetime.now().strftime(
        "%Y-%m-%d_%H_%M_%S_%f"
    )  ## this creates a timestamp to be used in the branch name
    original_branch = repo.active_branch  ## this gets the active branch

    # TODO Create more than one branch
    new_branch_name = "gothub_gots" + time  ## this creates a new branch name
    new_branch = repo.create_head(new_branch_name)  ## this creates a new branch
    new_branch.checkout()  ## this checks out the new branch

    # Replace this with the actual code
    repo.git.commit(
        "--allow-empty", "-m", "empty commit: start"
    )  ## this creates an empty commit
    answer = one_branch_mrkl(inp)  ## this runs the one_branch_mrkl function
    repo.git.commit(
        "--allow-empty", "-m", "empty commit: end"
    )  ## this creates an empty commit

    original_branch.checkout()  ## this checks out the original branch

    return WriteRepoOut(
        new_branches=[new_branch],
        answer=answer,
    )  ## this returns the answer
