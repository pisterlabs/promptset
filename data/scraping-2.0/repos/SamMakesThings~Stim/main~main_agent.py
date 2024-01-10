import os

from agent_protocol import Agent
from dotenv import load_dotenv
from langchain.agents import ZeroShotAgent, AgentExecutor
# Import all the langchain bullshit
from langchain.agents import load_tools
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from utils import (
    read_chat_history,
    read_topic_batches,
    discard_topic,
    add_topic_to_chat_history,
)

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

tools = load_tools([], llm=llm)

# TODO: Write tools. Should mostly be wrappers of existing functions.
# Tools needed:
# Insert topic into conversation
tools.append(
    Tool.from_function(
        add_topic_to_chat_history,
        name="add_topic_to_chat_history",
        description=(
            "Tool for adding a topic to the chat history. Useful if the topic is new"
            " and needs to be surfaced to the user."
        ),
    )
)

# Remove batched info from topics (if already sent to user)
tools.append(
    Tool.from_function(
        discard_topic,
        name="discard_topic",
        description=(
            "Tool for the removal of a topic from the database. Useful if the topic is"
            " no longer going to be talked about or has significatnly low relevance."
        ),
    )
)
# TODO: Tool descriptions should include info on when something should
# ...be surfaced to a user


# tools.append(query_menoresources_tool)
tool_names = [tool.name for tool in tools]

prefix = """You are a generally capable agent whose specialty
is juggling multiple priorities. Your job is to take in info
from multiple sources, then decide how to act on that info."""

format_instructions = f"""Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {', '.join(tool_names)}. If you want to give a final answer, just use that, don't say Action: None
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""  # noqa: E501

suffix = """Make sure you start every line with one of Thought:, Action:, Action Input:, Observation:, or Final Answer:.

Topics to consider raising to the user:
{batched_topics_str}

Current conversation:
{chat_history_str}

"""  # noqa: E501

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    format_instructions=format_instructions,
    input_variables=[
        "batched_topics_str",
        "chat_history_str",
    ],
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
)

agent = ZeroShotAgent(
    llm_chain=llm_chain,
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
)


def run_main_agent():
    """Runs the main agent itself"""

    # Assemble prompt
    batched_topics_str = read_topic_batches()
    chat_history_str = read_chat_history()

    input_dict = {
        "batched_topics_str": batched_topics_str,
        "chat_history_str": chat_history_str,
    }

    # Run main agent
    agent_executor.run(input_dict)

    return None


def main():
    from app import task_handler, step_handler

    Agent.setup_agent(task_handler, step_handler).start(port=8001)


if __name__ == "__main__":
    load_dotenv()
    main()
