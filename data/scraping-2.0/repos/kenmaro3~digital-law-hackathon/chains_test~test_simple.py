import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(
    input_variables=["job"],
    template="{job}に一番オススメのプログラミング言語は何?"
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain("データサイエンティスト"))

