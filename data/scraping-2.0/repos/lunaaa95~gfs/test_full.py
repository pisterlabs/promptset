import os
os.environ['http_proxy'] = 'http://10.177.27.237:7890'
os.environ['https_proxy'] = 'http://10.177.27.237:7890'

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from prompt.chain import chain_factory
from prompt.template import news_summary_prompt, news_predict_prompt
from answer.parser import GeneralAnswerParser
from prompt.rule import TrendRule, VolatilityRule, ActiveLevelRule
api_key = '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'


if __name__ == '__main__':
    openai = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=api_key)

    gap = GeneralAnswerParser([TrendRule, VolatilityRule, ActiveLevelRule])

    fpc = chain_factory(openai, gap)

    trend = '下跌'
    volatility = '高'
    active_level = '低'
    title = '金融精准发力,赋能实体经济'
    body = """加快实现高水平科技自立自强是推动高质量发展的必由之路。代表委员认为，资本市场作为连接实体经济、金融、科技的重要枢纽，需不断完善多层次市场体系，提升对科技创新的服务功能，推动科技、产业和金融高水平循环。应大力发展科创基金、政府引导基金等，加大对科技创新型企业的金融支持。
    引领创新驱动发展是资本市场肩负的重要历史使命。纵观海外创新型经济体的发展史，背后往往有着强大的多层次资本市场体系的支持。
    随着多层次资本市场体系不断完善，量身定制的发行上市、再融资、并购重组、股权激励等创新制度，为科技创新企业成长提供肥沃土壤，汇聚起一批涉及各产业链环节、多应用场景的创新企业，促进科技、资本和产业高水平循环更进一步。
    科创基金是聚焦科技创新主题的投资基金，也是加快实现高水平科技自立自强的重要力量。建议大力发展科创基金，加快实现高水平科技自立自强。全国人大代表、北京证监局局长贾文勤表示，为推动科创基金在我国加快实现高水平科技自立自强中发挥更大作用，应针对性地解决科创基金面临的痛点、难点问题。
    """

    result = fpc.run(
        trend=trend,
        volatility=volatility,
        active_level=active_level,
        title=title,
        body=body
        )
    
    print(result)
