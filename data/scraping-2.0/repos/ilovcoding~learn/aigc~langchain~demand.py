import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI()

promptZhCN = PromptTemplate(
    input_variables=["lang","demand"],
    template="用 {lang} 实现 {demand}",
)


zhCN_Chain = LLMChain(llm=llm, prompt=promptZhCN)
print(zhCN_Chain.run(lang="JS", demand="选择排序"))

