# Derived from the example in the docs, https://python.langchain.com/docs/langserve
#
# Remember to set the OPENAI_API_KEY environment variable to an OpenAI API key.
#
# You can try the joke generator by running this file and then visiting http://localhost:8000/joke/playground/
# The server API documentation is available at http://localhost:8000/docs
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
