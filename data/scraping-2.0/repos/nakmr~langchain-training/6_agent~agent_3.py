from dotenv import load_dotenv
load_dotenv()

import random
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.file_management import WriteFileTool

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

tools = []

tools.append(
    WriteFileTool(
        root_dir="./"
    )
)

def min_limit_random_number(min_number):
    return random.randint(int(min_number), 100000)

tools.append(
    Tool(
        name="Random",
        description="Generate a random number greater than or equal to a specific minimum value.",
        func=min_limit_random_number
    )
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("10以上のランダムな数字を生成した後、random.txtというファイルに保存してください。")
print(f"実行結果: {result}")