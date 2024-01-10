from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

import streamlit as st
few_shots = {
    "비트코인 가격": "SELECT close as 종가 FROM btcusdt_kline_1d order by time desc limit 2",   
}
few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]



embeddings = OpenAIEmbeddings()

few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]
vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the user question.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)
custom_tool_list = [retriever_tool]

from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
HOST=st.secrets["host"]
PORT=st.secrets["port"]
USER=st.secrets["user"]
PASSWORD=st.secrets["password"]
DATABASE=st.secrets["database"]
db = SQLDatabase.from_uri(f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}')
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

custom_suffix = """
I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
json 형태로 답변해줘
{
"respone": "I have obtained the closing prices of Ethereum from the "ethusdt_kline_1d" table. The most recent closing price is 2051.76. 
Final Answer: The current price of Ethereum is 2051.76."
}
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
)
# agent = create_sql_agent(
#     llm=llm,
#     toolkit=SQLDatabaseToolkit(db=db, llm=llm),
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

respontse =agent.run("이더리움 가격")

st.write(respontse)

