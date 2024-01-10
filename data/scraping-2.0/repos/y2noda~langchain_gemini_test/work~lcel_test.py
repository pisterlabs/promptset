# %%
import os
from operator import itemgetter

from google.cloud import aiplatform
from langchain.chat_models import ChatVertexAI
from langchain.globals import set_debug
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableMap
from langchain.sql_database import SQLDatabase

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing-private"
set_debug(True)

gcp_project_id = os.getenv("GCP_PROJECT_ID")
aiplatform.init(project=gcp_project_id)

gcp_dataset_id = os.getenv("GCP_DATASET_ID")
sqlalchemy_uri = f"bigquery://{gcp_project_id}/{gcp_dataset_id}"

db = SQLDatabase.from_uri(sqlalchemy_uri)

template = """Based on the table schema below, write a SQL query that would answer the user's question.
Please adhere to the syntax for SQL queries in GoogleSQL for BigQuery.:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)


model = ChatVertexAI(
    model="gemini-pro",
    max_output_tokens=2048,
    temperature=0,
    top_p=1,
    # top_k=40,
    verbose=True,
)


def get_schema(_):
    return db.get_table_info()


def cleaned_query(sql_query):
    cleaned_query = sql_query.strip().removeprefix("```sql\n").removesuffix("\n```")
    return cleaned_query


inputs = {"schema": RunnableLambda(get_schema), "question": itemgetter("question")}

sql_response = (
    RunnableMap(inputs)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
    | cleaned_query
)

result = sql_response.invoke(
    {
        "question": """\
## 命令
アプリユーザーのデータを抽出するためのSQLクエリを生成してください。クエリの目的は、プッシュ通知が配信された日から1週間以内にゲスト会員からアプリ本会員になったユーザーを見つけることです。
制約条件は必ずすべて守ってください。
SQLを生成する際には、SQL生成例を参考にしてください。

## 条件
- DatasetName は marketing_sample_data とします。
- ProjectName は langchain-gemini-test とします。
- プッシュ配信日は、"2023-11-15"として新たに定義します。データ型はDATE型です。


## 制約条件
- TIMESTAMP型のカラムは、DATE型に必ず変換して、使用してください。
- 全てのテーブルの created_at のカラムのデータ型はTIMESTAMP型です。必ずDATE型に変換してください。
- データ型に関するエラーが発生しないように、必ずデータ型を確認してください。
- 異なるデータ型をWHERE句などの条件としてで使用する場合は、必ず変換してください。
- TIMESTAMP型のカラムをWHERE句などの条件として使用する場合は、必ずDATE型に変換して使用してください。
- エイリアスを定義した場合は、後のクエリで必ずエイリアスを使用してください。
- CTEを利用して、複雑なクエリを分割して、読みやすく、管理しやすいクエリにしてください。
- アプリユーザーには、customer_no に値を持つアプリ本会員と customer_no が null のゲスト会員がいます。
- プッシュ通知日以前に会員になったユーザーを、ターゲットユーザーとして抽出するCTEを作成してください。
- ターゲットユーザーの中から、プッシュ通知が配信された日から1週間以内にゲスト会員からアプリ本会員になったユーザーを抽出するCTEを作成してください。
- 最終的に抽出するカラムは、カード番号、プッシュ配信日（2023-11-15）、本会員登録日です。
- エイリアスは必ず英語にしてください。
- テーブルは\`[プロジェクト名].[データセット名].[テーブル名]\`で指定すること。
- クエリ生成時はBigQueryの記法に従ってください。
- 異なるデータ型の場合、キャストをしてデータ型を揃えてください。
- サブクエリがスコープ外にあるとき、直接参照しないでください。
- SELECT句が3カラム以上の場合、並び替えは日付と名前で行ってください。
- SELECT句に存在しないカラムのORDER BYは除外してください。
- ORDER BYには、関数は含めないでください。
- 命令に対して適切なカラムで並び替えを行うこと。
- 集計関数を用いたカラムに対して必ずカラム名を付与すること。


## SQL生成例
```sql
-- 複数のCTEを定義し、テーブルにエイリアスを使用
WITH FirstCTE AS (
  SELECT
    FT.columnA,
    FT.columnB
  FROM
    your_dataset.first_table AS FT
  WHERE
    FT.condition1 = 'value1'
),
SecondCTE AS (
  SELECT
    ST.columnC,
    ST.columnD
  FROM
    your_dataset.second_table AS ST
  WHERE
    ST.condition2 = 'value2'
)

-- CTEを組み合わせて使用
SELECT
  f.columnA,
  f.columnB,
  s.columnC,
  s.columnD
FROM
  FirstCTE f
JOIN
  SecondCTE s ON f.columnA = s.columnC
```
"""
    }
)
print(result)

# %%

res_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(res_template)

full_chain = (
    RunnableMap(
        {
            "question": itemgetter("question"),
            "query": sql_response,
        }
    )
    | {
        "schema": RunnableLambda(get_schema),
        "question": itemgetter("question"),
        "query": itemgetter("query"),
        "response": lambda x: db.run(x["query"]),
    }
    | prompt_response
    | model
)

result = full_chain.invoke({"question": "How many users are there?"})
print(result)

# %%
