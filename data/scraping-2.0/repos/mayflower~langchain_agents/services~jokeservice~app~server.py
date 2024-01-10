from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI

app = FastAPI(
    title="JokeServe",
    description="A joke server that uses GPT-3.5-turbo to generate jokes.",
    version="0.1.0",
)

# openai transparent default route to openai
add_routes(app, 
           AzureChatOpenAI(
                deployment_name="GPT4",
           ),
            path="/openai",
           )

model = AzureChatOpenAI(
    deployment_name="GPT4",
)

# a joke telling route
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

add_routes(app,
           prompt | model,
           path="/chain",
           )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
