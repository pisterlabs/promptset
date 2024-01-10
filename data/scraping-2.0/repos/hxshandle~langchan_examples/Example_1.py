# // use langchain workwith openai
# // 2021-10-15
import os
from langchain.llms import OpenAI
from langchain.agents import load_tools, AgentType
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


def main():
    llm = OpenAI(model_name="text-davinci-003", max_tokens=1024)
    # 加载 serpapi 工具
    tools = load_tools(["serpapi"])
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    out = agent.run("""What's the date today? What great events have taken place today in history?""")
    print(out)


if __name__ == '__main__':
    main()
