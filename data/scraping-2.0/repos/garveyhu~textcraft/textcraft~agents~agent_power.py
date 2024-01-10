import textwrap
import time

from langchain.agents import AgentType, Tool, initialize_agent

from textcraft.models.llms.qwen import Qwen
from textcraft.tools.label_tool import LabelTool
from textcraft.tools.qa_tool import QATool
from textcraft.tools.title_tool import TitleTool

llm = Qwen(temperature=0)

# tools
tools = [
    Tool(
        name="title tool",
        func=TitleTool().run,
        description="Generate a title for the text.",
    ),
    Tool(
        name="label tool",
        func=LabelTool().run,
        description="Label the text.",
    ),
    Tool(
        name="vector database knowledge Q&A",
        func=QATool().run,
        description="Q&A with vector database knowledge base.",
    ),
]

# agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


def run_agent(input_text: str) -> str:
    response = agent.run(input_text)
    return response


def output_response(response: str) -> None:
    if not response:
        exit(0)
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # Add a delay of 0.1 seconds between each character
            print(" ", end="", flush=True)  # Add a space between each word
        print()  # Move to the next line after each line is printed
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    while True:
        try:
            input_text = input("question：")
            response = agent.run(input_text)
            output_response(response)
        except KeyboardInterrupt:
            break
