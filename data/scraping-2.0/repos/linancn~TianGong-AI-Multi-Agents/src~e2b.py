from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import E2BDataAnalysisTool

load_dotenv()

e2b_data_analysis_tool = E2BDataAnalysisTool(
    on_stdout=lambda stdout: print("stdout:", stdout),
    on_stderr=lambda stderr: print("stderr:", stderr),
)
e2b_data_analysis_tool.install_python_packages("pandas")

tools = [e2b_data_analysis_tool]

chat_model = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0,
)

agent = initialize_agent(
    tools=tools, llm=chat_model, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
)

response = agent.run(
    "Write a python code to count the number of words in the sentence: 'an educational assessment intended to measure the respondents' knowledge or other abilities'."
)

print(response)
