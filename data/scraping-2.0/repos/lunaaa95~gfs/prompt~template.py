from langchain import PromptTemplate

# 新闻总结Prompt
_news_template_chinese = """你正在逐段落概括长篇新闻.
你对上文的概括:{preceding}
你正在阅读的段落:{reading}
正在阅读的段落概括为?
"""

_news_template_english = """You are summarizing news paragraph by paragraph.
Your preceding summary:{preceding}
Current paragraph you are reading:{reading}
Summary current paragraph briefly as?
"""

news_summary_prompt = PromptTemplate(
    template=_news_template_chinese,
    input_variables=['preceding', 'reading']
)


# ICL Prompt新闻预测
_prediction_template_chinese = """你是一位阅读过大量中文新闻的诚实的股票分析师.你会在当前价格、当前波动率和当前交易活跃程度下判断新闻的影响.'
例子:
当前价格上涨,当前波动率高,当前交易活跃程度中等,新闻是复旦复华昨日公告,公司注意到复旦大学研发的类ChatGPT模型MOSS引发了大量关注和讨论,为避免相关信息对投资者造成误导,现予以澄清说明。截至目前,公司未参与复旦大学相关研究团队的研发工作,相关事项与公司不存在任何关系。目前,复旦大学与公司不存在股权关系//预测价格下跌,预测波动率高,预测交易活跃程度高.
当前价格震荡,当前波动率低,当前交易活跃程度低,新闻是今年全国两会,政府工作报告明确提出“大力发展数字经济”。实际上,全国两会召开之前,中共中央、国务院已经印发《数字中国建设整体布局规划》,提出全面提升数字中国建设的整体性、系统性、协同性,促进数字经济和实体经济深度融合。而随着数字中国建设加速推进,扮演我国算力基础设施和骨干传输网络建设者角色的三大运营商也将迎来新的发展期。//预测价格震荡,预测波动率低,预测交易活跃程度低.
当前价格下跌,当前波动率低,当前交易活跃程度中等,新闻是2022年,宁德时代实现营业收入3285.94亿元,同比增长152.07%;归母净利润307.29亿元,同比增长92.89%。成为中国汽车及零部件上市公司中最“赚钱”的公司。//预测价格上涨,预测波动率低,预测交易活跃程度中等.
现在你要回答:
当前价格{trend},当前波动率{volatility},当前交易活跃程度{active_level},新闻是{news}//预测价格、预测波动率、预测交易活跃程度分别是?"""

news_predict_prompt = PromptTemplate(
    template=_prediction_template_chinese,
    input_variables=['trend', 'volatility', 'active_level', 'news']
)


# 纯Prompt新闻预测
_template = """你是一位阅读过大量中文新闻的诚实的股票分析师,同时你会考虑股票的价格趋势,波动率和交易活跃程度.
股票的价格趋势包括:
- 下降
- 上涨
- 震荡

股票的波动率包括:
- 低
- 高

股票的交易活跃程度包括:
- 高
- 中等
- 低

记住当前股票的价格趋势是{trend},波动率是{volatility},交易活跃程度是{active_level},并且{delivery_day}、{holiday},再阅读新闻:{news}
"""

def template_factory():
    return PromptTemplate(
        input_variables=['trend', 'volatility', 'active_level', 'delivery_day', 'holiday', 'news'],
        template=_template,
    )
