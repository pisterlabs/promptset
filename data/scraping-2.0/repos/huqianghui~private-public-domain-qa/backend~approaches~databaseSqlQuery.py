import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines
import json
import mysql.connector

class DatabaseSqlQueryApproach(Approach):

    sql_template = """SELECT 
{sql}
"""

    prompt_prefix = """### MySQL SQL tables, with their properties:
#
# 股票(序号, 代码, 名称, 相关链接, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 今开, 昨收, 量比, 换手率,市盈率,市净率)
#
# 表名称也是中文
# 查询结果中，只显示代码，名称，以及必须要的字段
### {question}

SELECT
"""

    query_prompt_template = """以下是到目前为止的对话历史，以及用户提出的一个新问题。
    根据对话和新问题生成查询条件。
    请勿在搜索查询词中包含引用的源文件名和文档名称，例如信息.txt或文档.pdf。
    不要在搜索查询词的 [] 或<<>>中包含任何文本。

问题:
{question}

查询条件:
"""

    def __init__(self, gpt_deployment: str, codex_deployment: str):
        self.codex_deployment = codex_deployment
        self.gpt_deployment = gpt_deployment

    def run(self, question: str, overrides: dict) -> any:
        # STEP 1: Generate an optimized keyword search query based on the question
        prompt = self.query_prompt_template.format(question=question)
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=500, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text

        # STEP 2: generate sql from codex deployment
        prompt = self.prompt_prefix.format(question=q)
        completion = openai.Completion.create(
            engine=self.codex_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=1500, 
            n=1, 
            stop=[";"])
        sql = self.sql_template.format(sql=completion.choices[0].text)

        # STEP 3: Generate a response to the user based on the sql query
        mydb = mysql.connector.connect(
        host="XXXX.mysql.database.azure.com",
        user="XXX",
        password="XXX",
        database="XXX"
    )
        mycursor = mydb.cursor()
        mycursor.execute(sql)
        row_headers=[x[0] for x in mycursor.description] 
        print(row_headers)
        myresult = mycursor.fetchall()
        json_data=[]
        for result in myresult:
            json_data.append(dict(zip(row_headers,result)))
        print(json_data)
        return {"data_points": sql, "answer": json_data, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}        


        
