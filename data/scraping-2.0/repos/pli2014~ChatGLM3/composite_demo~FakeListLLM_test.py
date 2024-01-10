from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain_experimental.utilities import PythonREPL
from chatglm3_6b_llm import Chatglm3_6b_LLM

llm = Chatglm3_6b_LLM(n=10)

# 定义函数系列工具
python_repl = PythonREPL()
# 函数定义-python
def get_python_repl(script:str)->str:
    """
        1, 不做安全验证
    """
    print(f"call this script:{script}")
    return python_repl.run(script)


# 生成工具类
python_repl_tool = Tool(
    name="python_repl_tool",
    description="这是一个调用执行Python脚本",
    func=get_python_repl
)

llm_math = LLMMathChain(llm = llm)

# initialize the math tool
llm_math_tool = Tool(
    name ='llm_math_tool',
    func = llm_math.run,
    description ='Useful for when you need to answer questions about math.'
)


tools = [python_repl_tool, llm_math_tool]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, max_iterations=5, early_stopping_method="generate")

user_input = "执行下python脚本: print('hello world')"
answer = agent.run(user_input)
print(answer)
