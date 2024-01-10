import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["user", ],
    template="What is a good name for a company that makes {product}?",
)

OpenAI.openai_api_key = os.environ["OPENAI_API_KEY"]

# 定义模型和相关参数
llm = OpenAI(temperature=0.9)

text = "中国的房地产限售政策，会对房地产的泡沫有保护作用吗?"
print(llm(text))

prompt.format(product="colorful socks")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('————————————————————————')
    chain = LLMChain(llm=llm, prompt=prompt)
    chain.run("")