import pymysql
import os

from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate


def query_loan_chain(chat: str) -> str:
    if chat is None:
        return ""

    # chat = global_chat
    print(f"chat : {chat}")

    db = pymysql.connect(
        host="j9a405.p.ssafy.io",
        port=3306,
        user="root",
        passwd=f"{os.environ.get('MYSQL_PASSWORD')}",
        db="loan",
        charset="utf8",
        autocommit=True,
    )

    db = SQLDatabase.from_uri(f"mysql+pymysql://root:{os.environ.get('MYSQL_PASSWORD')}@j9a405.p.ssafy.io:3306/loan",
                              include_tables=["mortgage_loan", "jeonse_loan", "credit_loan"],
                              sample_rows_in_table_info=5)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Use the following format. SQLQuery는 유저에게 최대한 많은 정보를 제공하기 위해, id를 제외한 모든 column을 포함하여 조회하게끔 생성되어야만 합니다.:

        Question: "Question here"
        SQLQuery: "SQL Query to run (SELECT * FROM table_name WHERE conditions)"
        SQLResult: "Result of the SQLQuery"
        Answer: Final answer with SQL Result in JSON format as below.
        {{
          "key1": "value1",
          "key2": "value2",
          "key3": "value3",
          ...,
          "keyN": "valueN"
        }}

        Only use the following tables:

        {table_info}

        If someone asks for the table credit_loan(개인신용대출), 최저금리를 조회하기 위해 사용되는 column 이름을 신용점수에 따라 적절히 선택하여야 한다.

        Question: {input}
        """

    PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )

    # db_chain = SQLDatabaseChain.from_llm(
    #     llm, db, prompt=PROMPT, verbose=True, use_query_checker=True, return_intermediate_steps=True
    # )

    db_chain = SQLDatabaseChain.from_llm(
        llm, db, prompt=PROMPT, verbose=True, use_query_checker=True
    )

    response = db_chain.run(chat)
    # response = db_chain(chat)
    # return response["intermediate_steps"]
    return response