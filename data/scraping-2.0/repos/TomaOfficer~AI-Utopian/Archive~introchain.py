import os
from langchain import OpenAI
from langchain.agents import Tool, load_tools, initialize_agent
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=openai_api_key,
             temperature=0,
             model_name="text-davinci-003")

prompt = PromptTemplate(input_variables=["query"], template="{query}")

llm_chain = LLMChain(llm=llm, prompt=prompt)

# initialize the LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic')

# llm_math = LLMMathChain(llm=llm)

# # initialize the math tool
# math_tool = Tool(
#     name='Calculator',
#     func=llm_math.run,
#     description='Useful for when you need to answer questions about math.')

# tools = load_tools(['llm-math'], llm=llm)

# zero_shot_agent = initialize_agent(agent="zero-shot-react-description",
#                                    tools=tools,
#                                    llm=llm,
#                                    verbose=True,
#                                    max_iterations=3)

# zero_shot_agent("what is (4.5*2.1)^2.2?")
