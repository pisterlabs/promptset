import io
import threading
from typing import Any, Dict, List, Union
import os
import sys
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

import openai

from config import Config
from services.ThreadedGenerator import ChainStreamHandler, ThreadedGenerator


class ChatgptOriginal:
    def streamChat(question, collection_id):
        generator = ThreadedGenerator()
        threading.Thread(target=ChatgptOriginal.askQuestion, args=(generator, collection_id, question)).start()
        return generator
    
    def askQuestion(generator, collection_id, question):
        try:
            batch_messages = [
                [
                    SystemMessage(
                        content="."),
                    HumanMessage(
                        content="今天怎么样")
                ]
            ]
            openai.api_key = Config.OPENAI_API_KEY

            chat = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager(
            [ChainStreamHandler(generator)]), verbose=True)

            result = chat.generate(messages=batch_messages)

            res_dict = {
            }

            res_dict["source_documents"] = []

            for source in result["source_documents"]:
                res_dict["source_documents"].append({
                    "page_content": source.page_content,
                    "metadata":  source.metadata
                })

            return res_dict

        finally:
            generator.close()

    def chat(question):
        batch_messages = [
            [
                SystemMessage(
                    content="You are a helpful assistant that translates English to Chinese."),
                HumanMessage(
                    content=question)
            ]
        ]

        chat = ChatOpenAI(temperature=0)

        result = chat.generate(batch_messages)
        print(result)
        print(result.llm_output['token_usage'])
        return result.generations[len(batch_messages)-1][0].text
