# to use agents there 3 basic components
# 1. the LLM
# 2. the tool where the agent is going to interact with
# 3. the agent itself

from langchain.agents import load_tools
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain import LLMMathChain
from langchain import OpenAI
# list of openai models
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()

# setup the llm
llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model_name="text-davinci-003"
)

# initialize the tool (LLMMathchain)

# setup the model
llm_math = LLMMathChain(llm=llm)

# initialize the math tool
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="A useful tool when trying to solve math problems"
)

# the tools must be passed as a list to the llm
tools = [math_tool]

# now we will initalize an agent to use the tool

# settingup agent
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

# lets try asking a calculation question
zero_shot_agent("what is (4.5*2.5)^2.2")

# trying on worded problems
zero_shot_agent("A car is traveling at 60 miles per hour for the first hour, then increases speed to 80 miles per hour for the remaining 30 minutes. what distance has it traveled?")

zero_shot_agent("Reece brings in four apples, and James brings in 2 boxes of apples."
                "Each box contains 12 apples. How many apples do they have in total?")

# what if we want the agent to answer general questions in addition to math questions

# setup the llm chain to obtain the prompt
prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# initalizing the LLM tool
llm_tool = Tool(
    name="Language Model",
    func=llm_chain.run,
    description="A useful tool when trying to answer general questions"
)

# lets add the new tool to the tools list
tools.append(llm_tool)

# now lets reinitialize the aget with both tools
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

# lets re-run the query asking about capital norway
zero_shot_agent("what is the capital of norway?")


# NOTE: langchain has already developed a set of pre-built tools

tools = load_tools(
    ['llm-math'],
    llm=llm
)

# check the tool
print(tools[0].name, '--->', tools[0].description)

# LangChain tutorial by James Briggs
