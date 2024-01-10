from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain

from langchain.prompts import PromptTemplate

template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。

网页是新浪财经A股上市公司的每季度股东信息表格。

请抽取参数请求的信息。每个截至日期作为JSON返回数据的date_of_quarter。因此，当表格中有多个截止日期时，返回数据应当包括所有的日期作为key。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "date_of_quarter": [
    {{
      "holder_name": "a",
      "percentage": "50"
    }},
    {{
      "holder_name": "b",
      "percentage": "30"
    }},
  ]
}} 

例如，截至日期为2023-03-31，JSON数据应该是如下形式:

{{
  "2023-03-31": [
    {{
      "holder_name": "a",
      "percentage": "50"
    }},
    {{
      "holder_name": "b",
      "percentage": "30"
    }},
  ]
}}
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["requests_result"],
    template=template,
)

chain = LLMRequestsChain(llm_chain=LLMChain(
    llm=OpenAI(temperature=0), prompt=PROMPT))

# question = "the lottery result of 2023-04-30"

inputs = {
    "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockHolder/stockid/600519/displaytype/30.phtm"
}

response = chain(inputs)
print(response['output'])
