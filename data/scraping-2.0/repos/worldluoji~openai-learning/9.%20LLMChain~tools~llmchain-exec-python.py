
import openai, os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

openai.api_key = os.environ.get("OPENAI_API_KEY")

multiply_by_python_prompt = PromptTemplate(template="请写一段Python代码，计算{question}?", input_variables=["question"])
math_chain = LLMChain(llm=llm, prompt=multiply_by_python_prompt, output_key="answer")
answer_code = math_chain.run({"question": "352乘以493"})

# LangChain 里面内置了一个 utilities 的包，里面包含了 PythonREPL 这个类，可以实现对 Python 解释器的调用。如果你去翻看一下对应代码的源码的话，它其实就是简单地调用了一下系统自带的 exec 方法，来执行 Python 代码。
from langchain.utilities import PythonREPL
python_repl = PythonREPL()
result = python_repl.run(answer_code)
print(result)

# LangChain 就把这个过程封装成了一个叫做 LLMMathChain 的 LLMChain。不需要自己去生成代码，再调用 PythonREPL，只要直接调用 LLMMathChain，它就会在背后把这一切都给做好

from langchain import LLMMathChain

llm_math = LLMMathChain(llm=llm, verbose=True)
result = llm_math.run("请计算一下352乘以493是多少?")
print(result)

# 其它： SQLDatabaseChain 可以直接根据你的数据库生成 SQL，然后获取数据，LLMRequestsChain 可以通过 API 调用外部系统，获得想要的答案
# https://python.langchain.com/en/latest/modules/agents/tools.html