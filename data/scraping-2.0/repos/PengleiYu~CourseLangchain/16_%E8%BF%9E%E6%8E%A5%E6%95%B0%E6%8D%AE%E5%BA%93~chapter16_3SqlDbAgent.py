# 代理按ReAct风格进行推理，多次调用SQL工具查询信息，最终汇总结论
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms.openai import OpenAI
from langchain.utilities.sql_database import SQLDatabase

db = SQLDatabase.from_uri('sqlite:///FlowerShop.db')
llm = OpenAI(verbose=True)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

questions = ["哪种鲜花的存货数量最少？", "平均销售价格是多少？", ]
for q in questions:
    response = agent_executor.run(q)
    print(response)
