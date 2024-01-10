import calendar
import pandas as pd
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key="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",
    request_timeout=30,
    max_retries=1,
    streaming=True,
)
# 单个csv文件
df = pd.read_csv("./util/商业智能BI案例数据.csv")
agent = create_pandas_dataframe_agent(chat_model, df, verbose=True)
# 如果多个csv用这行代码
# agent = create_csv_agent(OpenAI(temperature=0), ['titanic.csv', 'titanic_age_fillna.csv'], verbose=True)

# 画图
# question = """
# 画一个柱状图, x轴是"门店", y轴是该"门店"的"数量"总和
# """
question = """
画一个柱状图, x轴是"订单日期"列所属的月份, 纵轴是"产品ID"=3001的"数量"的总和. 图表上的各种信息为中文
"""
agent.run(question)
