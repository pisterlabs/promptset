from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.tools import StructuredTool
from langchain_community.chat_models import ChatOpenAI

from src.agents import agent
from src.github import issues
from src.utils import common

# Load Env Vars
_ = load_dotenv(find_dotenv())

# Load Config
CONFIG = common.config()

# Initialize Logger
logger = common.create_logger(__name__)

# Constants
MODEL = CONFIG["openai"]["model"]
TEMPERATURE = CONFIG["openai"]["temperature"]
PM_PROMPT = (
    "Act as Project Manager, break down the following task and assign role "
    "(i.e.: Python Developer, Frontend Developer, etc.) to each subtask. "
    "Create an issue for each subtask, with the format: `{ROLE}_{TASK}`. "
    "Comment the result on GitHub."
)
PM_AGENT_USER_INPUT_TEMPLATE = "Number: {issue_number}\n\nTitle: {issue_title}\n\nDescription: {issue_body}"


def comment_on_github_issue(issue_number: int, comment: str) -> str:
    """Agent Tool to comment on a GitHub Issue.

    Args:
        issue_number (int): The issue number to comment on.
        comment (str): The comment to post on the issue.

    Returns:
        str: A confirmation of the action.
    """
    issues.create_comment_on_issue(issue_number, comment)
    return "Commented on GitHub Issue"


def create_issue_on_github(title: str, body: str) -> str:
    """_summary_

    Args:
        title (str): The title of the issue.
        body (str): The body of the issue.

    Returns:
        str: A confirmation of the action.
    """
    issues.create_issue(title, body)
    return "Created a GitHub Issue"


PM_AGENT = agent.Agent(
    llm=ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE),
    tools=[StructuredTool.from_function(comment_on_github_issue), StructuredTool.from_function(create_issue_on_github)],
    prompt=PM_PROMPT,
    type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

if __name__ == "__main__":
    logger.info("Getting Issue...")
    issue = issues.get_issues()[0]

    logger.info("Formatting Issue Response...")
    user_input = PM_AGENT_USER_INPUT_TEMPLATE.format(
        issue_number=issue["number"],
        issue_title=issue["title"],
        issue_body=issue["body"],
    )

    logger.info("Invoking PM Agent Chain...")
    PM_AGENT.invoke(user_input)
