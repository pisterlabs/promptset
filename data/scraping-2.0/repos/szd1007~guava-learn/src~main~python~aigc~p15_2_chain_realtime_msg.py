import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import LLMChain
from  langchain.chains import SequentialChain
from langchain.chains import LLMRequestsChain
openai.api_key = os.environ.get("OPENAI_API_KEY")

#https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
#llm = OpenAI(model_name="gpt-4-1106-preview", max_tokens=2048, temperature=0.5)
llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

template = """在 >>> 和 <<< 直接是来自Google的原始搜索结果.
请把对于问题 '{query}' 的答案从里面提取出来， 如果没有相关信息的话就说 "找不到"
请使用如下格式：
Extracted:<answer or "找不到">
>>> {requests_result} <<<
Extracted:
"""
PROMPT = PromptTemplate(
    input_variables=['query', "requests_result"],
    template=template
)
requests_chain =  LLMRequestsChain(llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=PROMPT))
question = "今天北京的天气怎么样"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}
result = requests_chain(inputs)
print(result)
print(result['output'])