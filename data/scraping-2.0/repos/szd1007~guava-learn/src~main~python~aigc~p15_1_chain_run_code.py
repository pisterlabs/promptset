import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import LLMChain
from  langchain.chains import SequentialChain

openai.api_key = os.environ.get("OPENAI_API_KEY")

#https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
#llm = OpenAI(model_name="gpt-4-1106-preview", max_tokens=2048, temperature=0.5)
llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)
multiply_by_python_prompt = PromptTemplate(template="请写一段可运行的python代码，计算{question}?", input_variables=["question"])
match_chain = LLMChain(llm=llm, prompt=multiply_by_python_prompt, output_key="answer")
answer_code = match_chain.run({"question": "352乘以493"})
print(answer_code)

from langchain.utilities import PythonREPL

python_repl = PythonREPL()
result = python_repl.run(answer_code)
print(result)