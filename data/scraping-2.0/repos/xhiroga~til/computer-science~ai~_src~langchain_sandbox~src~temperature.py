from dotenv import load_dotenv
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI

load_dotenv()


llm = OpenAI(temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("昨日の東京の最高気温は華氏で何度だったでしょう？その数字、華氏から摂氏を引いた値は何ですか？")
