import os
import typer
from rich import print
from rich.prompt import Prompt
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from utils.api import HelpMeFindLawClient
from utils.tools import HelpMeFindlLawCompletionTool
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HMFL_API_KEY")

client = HelpMeFindLawClient(token=token)
tools = [
    HelpMeFindlLawCompletionTool(client=client),
    DuckDuckGoSearchRun()
]

model = ChatOpenAI(model_name="gpt-4", verbose=True)
planner = load_chat_planner(llm=model)
executor = load_agent_executor(llm=model, tools=tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

def main():
    prompt = Prompt.ask("\n\n[bold pink]Enter a prompt: [bold pink]")
    output = agent({"input": prompt})
    print(output)
    

if __name__ == "__main__":
    typer.run(main)
