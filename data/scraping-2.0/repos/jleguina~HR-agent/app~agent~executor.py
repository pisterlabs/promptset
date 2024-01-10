from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.tools import BaseTool

from app.agent.output_parser import CustomJSONOutputParser
from app.agent.tools import get_all_tools
from app.config import settings

TEMPLATE_TOOL_RESPONSE = """TOOL RESPONSE:
---------------------
{observation}

USER'S INPUT
--------------------

Okay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what!"""


def init_agent_executor(tools: list[BaseTool], verbose: bool = False) -> AgentExecutor:
    prompt = load_prompt(Path("./app/prompts/master.yaml").resolve())
    tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    prompt = prompt.partial(
        date=datetime.now().isoformat()[:10],
        tool_names=tool_names,
        tool_strings=tool_strings,
    )

    llm = ChatOpenAI(temperature=0.1, model=settings.OPENAI_MODEL)
    llm_with_stop = llm.bind(stop=["\nObservation"])

    # Using LCEL
    agent: Any = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_messages(
                x["intermediate_steps"],
                template_tool_response=TEMPLATE_TOOL_RESPONSE,
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_stop
        | CustomJSONOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    return agent_executor


if __name__ == "__main__":
    chat_history = []
    tools = get_all_tools()
    agent_executor = init_agent_executor(tools, verbose=True)

    init_message = "Hi, I am Maria, your personal HR assistant. To get started, can you please tell me your name and email address? Thanks!"
    print(init_message)
    chat_history.append({"role": "assistant", "content": init_message})

    while True:
        user_input = input(">>> ")
        out = agent_executor.invoke(
            {"input": user_input, "chat_history": chat_history}
        )["output"]
        print("Agent:", out)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": out})
