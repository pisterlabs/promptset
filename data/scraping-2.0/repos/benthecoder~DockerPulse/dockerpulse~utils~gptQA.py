import openai
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import re

# Load .env file
load_dotenv()


class GPTQA:
    def __init__(self):
        self.prompt = "I have a problem with my Docker logs. Can you help me troubleshoot?"
        self.temperature = 0.5
        self.max_tokens = 2048
        self.model_engine = "gpt-3.5-turbo"
        self.llm = ChatOpenAI(model_name=self.model_engine,
                              temperature=self.temperature, max_tokens=self.max_tokens)

    def search(query):
        with DDGS() as ddgs:
            for r in ddgs.text(query):
                return r

    def generate_solution(self, logs):
        _, logs = logs
        tools = [
            Tool(
                name="search",
                func=self.search,
                description="Useful when you need to search for something on the internet.",
            ),]
        # Clean up the logs to remove any sensitive information
        logs = re.sub(r"([0-9a-fA-F]{64})", "<HASH>", logs)
        logs = re.sub(r"([0-9]{1,3}\.){3}[0-9]{1,3}", "<IP_ADDRESS>", logs)

        # Generate a solution using OpenAI's GPT-3 API
        prompt = f"{self.prompt}\n\nLogs:\n{logs}\n\nSolution:"

        agent = initialize_agent(
            tools, self.llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
        answer = agent.run(prompt)
        return answer


if __name__ == "__main__":
    gpt = GPTQA()
    print(gpt.generate_solution(
        "I have a problem with my Docker logs. Can you help me troubleshoot?"))
