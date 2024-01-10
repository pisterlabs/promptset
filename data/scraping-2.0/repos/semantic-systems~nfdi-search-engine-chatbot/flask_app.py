# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory

from nfdichat.common.config import (dataset_config, llm_config, main_config,
                                    retriever_config)
from nfdichat.datasets import *
from nfdichat.llms import *
from nfdichat.retrievers import *

app = Flask(__name__)

RETRIEVER: Retriever = eval(retriever_config[main_config["RETRIEVER"]])(
    document_processor=eval(
        dataset_config[main_config["DATASET"]]["document_processor"]
    )()
)
LLM_MODEL: LLM = eval(llm_config[main_config["LLM"]]["MODEL"])()


@app.route("/ping")
def ping():
    return "This is NFID-Search ChatBot"


@app.route("/chat", methods=["POST", "GET"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question")
    chat_history_list = data.get("chat-history")
    search_results = data.get("search-results")
    query, items = NFDISearchDataset().fetch(**{"results": search_results})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for chat_history in chat_history_list:
        memory.save_context(
            {"input": chat_history["input"]}, {"output": chat_history["output"]}
        )
    global LLM_MODEL, RETRIEVER
    CHATBOT = ConversationalRetrievalChain.from_llm(
        llm=LLM_MODEL,
        chain_type="stuff",
        retriever=RETRIEVER.build_retriever(docs=items),
        memory=memory,
    )
    answer = CHATBOT({"question": question})
    answer = answer["answer"]
    chat_history_list.append({"input": question, "output": answer})
    data["chat-history"] = chat_history_list
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
