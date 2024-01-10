import os
os.environ['http_proxy'] = 'http://10.177.27.237:7890'
os.environ['https_proxy'] = 'http://10.177.27.237:7890'

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from prompt.chain import NewsPredictChain
from prompt.template import news_predict_prompt
api_key = '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'


if __name__ == '__main__':
    openai = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=api_key)
    llm_chain = LLMChain(prompt=news_predict_prompt, llm=openai)

    npc = NewsPredictChain(chain=llm_chain)

    trend = '下跌'
    volatility = '高'
    active_level = '低'
    news = '金融精准发力，服务质效不断提升，赋能实体经济成效显著。在助力科技自立自强方面，试点注册制以来，资本市场对科技创新的服务功能进一步增强，资本“活水”加速流入科技创新行业。'

    test = npc.run(trend=trend, volatility=volatility, active_level=active_level, news=news)

    print(test)
