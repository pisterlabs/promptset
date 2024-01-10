import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('BACKEND_OPENAI_API_KEY')
template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
分析网页的HTML内容，这是一份贵州茅台的资产负债表，请帮我生成一份财务报表的分析，
并告诉我重点需要注意哪些点。该公司股票是否值得投资。
>>> {requests_result} <<<
"""
prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=OPENAI_API_KEY)
chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
  "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vFD_BalanceSheet/stockid/600519/ctrl/part/displaytype/4.phtml"
}

response = chain(inputs)
# 返回的对象中包括output和url两个字段
