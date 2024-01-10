import os

from flask import Flask, jsonify, request
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.prompts.chat import (
    PromptTemplate
)
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

openaiLLM = AzureOpenAI(
    azure_endpoint="https://openai-jez.openai.azure.com/",
    azure_deployment="GPT-35-turbo",
    model="GPT-35-turbo",
    api_version="2023-05-15"
)

embeddings = AzureOpenAIEmbeddings()

# load the vectorstore
db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
RETRIEVER = db.as_retriever()
    
QA = RetrievalQA.from_chain_type(
    llm=openaiLLM, 
    chain_type="stuff", 
    retriever=RETRIEVER, 
    return_source_documents=True
)

app = Flask(__name__)


@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    global QA
    user_prompt = request.form.get("user_prompt")
    print(f'User Prompt: {user_prompt}')
    if user_prompt:
        # print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
        help_request = "Eres un asesor. Ayuda al usuario. Si no sabes la respuesta di que no tienes la información." +\
                        f"\nUser:{user_prompt}"            
        res = QA(help_request)
        answer, docs = res["result"], res["source_documents"]

        prompt_response_dict = {
            "Prompt": help_request,
            "Answer": answer,
        }

        prompt_response_dict["Sources"] = []
        for document in docs:
            prompt_response_dict["Sources"].append(
                (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            )

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400


if __name__ == "__main__":
    app.run(debug=True, port=5110)