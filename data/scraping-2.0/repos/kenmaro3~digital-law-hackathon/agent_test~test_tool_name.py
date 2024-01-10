from langchain.llms import OpenAI
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.agents import load_tools


tool_names = ["python_repl"]
tools = load_tools(tool_names)

llm = OpenAI(model_name="text-davinci-003")
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

tools.append(
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
)
print(tools)
print(len(tools))
