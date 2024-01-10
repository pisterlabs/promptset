from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# MySQL 数据库连接 URI 的格式通常是 "mysql+pymysql://username:password@host:port/database"
# 替换下面的参数为您的 MySQL 数据库的实际参数
mysql_uri = "mysql+mysqlconnector://dbtest:123456@39.99.142.7:3306/botmart"

db = SQLDatabase.from_uri(mysql_uri)
llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 执行查询
db_chain.run("都有什么产品?")
