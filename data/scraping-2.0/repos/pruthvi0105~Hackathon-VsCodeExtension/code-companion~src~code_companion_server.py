import os
import time
from flask import Flask, request
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.memory import ChatMemoryBuffer

app = Flask(__name__)

# Sample data to simulate a database
data = []
load_dotenv()
chat_engine = None

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
# Use this to connect to Azure Open AI and train your model further
llm = AzureOpenAI(
    deployment_name="azure_davinci",
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1000,
)


@app.route("/init", methods=["GET"])
def load_data():
    dir_path = request.args.get("path")
    print(f"Loading data from: {dir_path}")
    docs = SimpleDirectoryReader(dir_path, required_exts=".cs", recursive=True).load_data()
    print(f'Total documents: {len(docs)}')
    index = VectorStoreIndex.from_documents(
        documents=docs,
        service_context=ServiceContext.from_defaults(
            llm=llm, embed_model=OpenAIEmbeddings()
        ),
    )

    global chat_engine
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt="You are a chatbot, able to have normal interactions, answert the question in the given context also help with some c# scripts..",
    )

    return "Success"


@app.route("/query", methods=["GET"])
def query_data():
    question = request.args.get("query")
    print(question)
    response = chat_engine.chat(f"{question}")
    return str(response)

@app.route("/shutdown", methods=["GET"])
def shutdown():
    time.sleep(2)
    os.kill(os.getpid(), 9)
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run(debug=True, port=5000)
