# 根据输入语句，模型返回SQL语句，执行结果后得出结论
from langchain.utilities.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri('sqlite:///FlowerShop.db')
llm = OpenAI(verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# response = db_chain.run('有多少种不同的鲜花？')
# response = db_chain.run("哪种鲜花的存货数量最少？")
# response = db_chain.run("平均销售价格是多少？")
# response = db_chain.run("从法国进口的鲜花有多少种？")
response = db_chain.run("哪种鲜花的销售量最高？")
print(response)
