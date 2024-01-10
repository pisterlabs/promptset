from operator import itemgetter

from fastapi import FastAPI
from langserve import add_routes
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

import settings

OPEN_AI_API_KEY = settings.OPEN_AI_API_KEY

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that answers questions about people's hometowns."),
    ("human", "{person}はどこの都市に住んでいる人ですか?結果は都市の名前のみにしてください。出力例：[city]")
])
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that answers questions about people's hometowns."),
    ("human", "{city}はどこの国の都市ですか? 言語は{language}で答えてください。")
])

chat_model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model_name="gpt-3.5-turbo")

chain1 = prompt1 | chat_model | StrOutputParser()

print(chain1)

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | chat_model
    | StrOutputParser()
)

# 2. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 3. Adding chain route
add_routes(
    app,
    chain1,
    path="/chain1",
)

add_routes(
    app,
    chain2,
    path="/chain2",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)