from langchain.agents.agent_toolkits import O365Toolkit
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

toolkit = O365Toolkit()
tools = toolkit.get_tools()
# print(tools)

llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run(
    "Could you search in my inbox folder and let me know if I have received any mails from a.jabbour with a max result of 3 and don't truncate?"
)

# agent.run(
#     "Create an email draft for me to edit of a letter from the perspective of a sentient parrot"
#     " who is looking to collaborate on some research with her"
#     " estranged friend, a cat. Under no circumstances may you send the message, however."
# )