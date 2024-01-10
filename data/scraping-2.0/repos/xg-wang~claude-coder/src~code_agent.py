import click
from typing import Any, Generator
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chat_models import ChatAnthropic
from src.util import setup_logging
from src.util import MODEL_NAME, MAX_TOKEN_TO_SAMPLE
from src.code_search_tool import CodeSearchTool
from src.code_interpreter_tool import CodeInterpreterTool
import logging


def initialize_claude_coder(repo_url: str) -> AgentExecutor:
    repo_name = repo_url.split("/")[-1]
    code_search_tool = CodeSearchTool(repo_name=repo_name, repo_url=repo_url)
    code_interpreter_tool = CodeInterpreterTool()
    tools = [code_search_tool, code_interpreter_tool]
    agent = initialize_agent(
        tools,
        ChatAnthropic(
            temperature=0, model=MODEL_NAME, max_tokens_to_sample=MAX_TOKEN_TO_SAMPLE
        ),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


def run_agent_qa(question: str, agent: AgentExecutor) -> Generator[dict[str, Any], None, dict[str, Any]]:  # type: ignore
    prompt = f"<Instruction>When you have the answer, always say 'Final Answer:'</Instruction>\n{question}"
    for step in agent.iter(prompt):
        # logging.info(f"Step: {step}")
        if output := step.get("intermediate_step"):
            action, value = output[0]
            # logging.info(f"action:\n{action.tool}")
            # logging.info(f"tool input:\n{action.tool_input}")
            # logging.info(f"value:\n{value}")
            yield step
        elif output := step.get("output"):
            # logging.info(f"Output: {output}")
            yield step


@click.command(
    help='Run the code agent. Example: poetry run python src/code_agent.py --repo "https://github.com/chroma-core/chroma" --query "How do I use chromadb?"'
)
@click.option("--repo", required=True, help="The URL of the repo.")
@click.option("--query", required=True, help="The query for LLM.")
@click.option(
    "--verbose", required=False, is_flag=True, default=False, help="More logging."
)
def main(repo, query, verbose):
    setup_logging(verbose)
    load_dotenv(find_dotenv())
    agent = initialize_claude_coder(repo)

    for step in run_agent_qa(query, agent):
        # logging.info(f"Step: {step}")
        if output := step.get("intermediate_step"):
            action, value = output[0]
            logging.info(f"action:\n{action.tool}")
            logging.info(f"tool input:\n{action.tool_input}")
            logging.info(f"value:\n{value}")
        elif output := step.get("output"):
            logging.info(f"Output: {output}")


if __name__ == "__main__":
    main()
