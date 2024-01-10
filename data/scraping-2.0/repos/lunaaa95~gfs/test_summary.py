import os
os.environ['http_proxy'] = 'http://10.177.27.237:7890'
os.environ['https_proxy'] = 'http://10.177.27.237:7890'

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from prompt.chain import NewsSummaryChain
from prompt.template import news_summary_prompt
api_key = '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'


if __name__ == '__main__':
    openai = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=api_key)
    llm_chain = LLMChain(prompt=news_summary_prompt, llm=openai)

    nsc = NewsSummaryChain(chain=llm_chain)

    title = '昨夜，美国硅谷银行轰然倒塌。'
    body = """硅谷银行的母公司硅谷金融集团（SVB Financial）受到关注。该股昨日暴跌，此前该集团周四宣布计划筹集超过20亿美元的资金以抵消债券销售损失，导致其股价一日暴跌逾60%。
美国联邦存款保险公司周五表示，硅谷银行已被加州监管机构关闭；硅谷银行拥有约2090亿美元资产；该银行是今年第一家破产的受保险机构。
联邦存款保险公司表示，它已通过其创建的名为圣克拉拉存款保险国家银行的新实体控制了该银行。该监管机构表示，该银行的所有存款均已转移至新银行。'
美国联邦存款保险公司称，硅谷银行的总部和所有分支机构将于2023年3月13日星期一重新开放。
加州金融保护与创新部门（DFPI）周五公告显示，根据加州金融法典第592条，它已经接管了硅谷银行，理由是流动性不足和资不抵债。DFPI指定联邦存款保险公司（FDIC）作为硅谷银行的接管方。
硅谷银行是一家州特许商业银行，也是位于圣克拉拉的联邦储备系统的成员，截至2022年12月31日，总资产约为2090亿美元，总存款约为1754亿美元。它的存款由联邦存款保险公司在适用的限额下提供联邦保险。
曾经是银行业中的宠儿，硅谷银行在宣布债券持有的大笔亏损和加强资产负债表的计划后，迅速崩溃，其股价暴跌，引发了广泛的客户提款。
在星期五上午取消了计划中的22.5亿美元股票销售后，硅谷银行的母公司SVB Financial Group正在竭力寻找买家。监管机构不愿等待。加利福尼亚州金融保护与创新部周五几个小时内关闭了该银行，并将其置于联邦存款保险公司的控制之下。
硅谷银行本周早些时候在存款大幅下降超出预期后出售资产时损失近20亿美元，令投资者感到惊讶的是，自那以后，该银行股价已经下跌超过80％，科技客户因对该银行的健康状况感到担忧而争相提款。"""

    test = nsc.run(title=title, body=body)

    print(test['summary'])
