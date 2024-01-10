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
WORKER_PROMPT = "Act as {role}. Your task is to complete the request given. Comment your detailed solution on GitHub."
WORKER_AGENT_USER_INPUT_TEMPLATE = (
    "Task: {task}\n\nNumber: {issue_number}\n\nTitle: {issue_title}\n\nDescription: {issue_body}"
)


def comment_on_github_issue(issue_number: int, comment: str) -> str:
    """Agent Tool to comment on a GitHub Issue.

    Args:
        issue_number (int): The issue number to comment on.
        comment (str): The comment to post on the issue.
    """
    issues.create_comment_on_issue(issue_number, comment)
    return "Commented on GitHub Issue"


WORKER_AGENT = agent.Agent(
    llm=ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE),
    tools=[StructuredTool.from_function(comment_on_github_issue)],
    prompt=WORKER_PROMPT,
    type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

if __name__ == "__main__":
    logger.info("Getting Issues...")
    agents = {}
    for issue in issues.get_issues():
        if issue["number"] == 1:
            # skip the first issue
            continue
        role, task = issue["title"].split("_")
        if role not in agents:
            agents[role] = []
        agents[role].append(
            WORKER_AGENT_USER_INPUT_TEMPLATE.format(
                task=task,
                issue_number=issue["number"],
                issue_title=issue["title"],
                issue_body=issue["body"],
            ),
        )

    for role, agent_inputs in agents.items():
        logger.info(f"Creating Worker Agent for {role}...")
        WORKER_AGENT.prompt = WORKER_PROMPT.format(role=role)
        for agent_input in agent_inputs:
            logger.info("Invoking Worker Agent Chain...")
            WORKER_AGENT.invoke(agent_input)
