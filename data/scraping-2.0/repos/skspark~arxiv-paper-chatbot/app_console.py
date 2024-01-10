# -*- coding: utf-8 -*-

import argparse
import json

import arxiv
import ast
import os
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import faiss
import openai
import threading
from queue import Queue

from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from langchain.memory.chat_memory import ChatMessageHistory

from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from py_pdf_parser.loaders import load_file

from pydantic import BaseModel

from arxiv_paper_chatbot.chatbot import PaperChatbot
from arxiv_paper_chatbot.parser import PaperParser
from arxiv_paper_chatbot.paper import Paper

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseChatMessageHistory


def select(*queues):
    combined = Queue(maxsize=0)

    def listen_and_forward(queue):
        while True:
            combined.put((queue, queue.get()))

    for queue in queues:
        t = threading.Thread(target=listen_and_forward, args=(queue,))
        t.daemon = True
        t.start()
    while True:
        yield combined.get()


class OpenAIChatHandler(BaseModel, BaseCallbackHandler):
    call_count_limit: int = 100
    call_count: int = 0

    def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        print("on new token")
        self.call_count += 1
        if self.call_count > self.call_count_limit:
            raise Exception("call count limit exceeded")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            **kwargs: Any) -> Any:
        pass


def load_paper(arxiv_paper_id: str) -> Paper:
    try:
        search_results = arxiv.Search(id_list=[arxiv_paper_id]).get()
        if not search_results:
            raise RuntimeError("arxiv paper not found")
        paper = next(search_results)
    except Exception as e:
        raise RuntimeError("cannot fetch arxiv paper", e)
    #    load / parse paper pdf
    pdf_file = paper.download_pdf(dirpath='/tmp/')
    try:
        return PaperParser().parse(load_file(pdf_file))
    except Exception as e:
        raise Exception("cannot load / parse paper", e)
    finally:
        if os.path.exists(pdf_file):
            os.remove(pdf_file)


def main():
    parser = argparse.ArgumentParser(description='chatbot arguments')
    parser.add_argument('--arxiv-paper-id', '-i', dest='arxiv_paper_id', help='paper arxiv id', default="2306.15577")
    parser.add_argument('--openai-api-key', '-k', dest='openai_api_key', help='openai api key')
    parser.add_argument('--llm-call-limit', '-l', dest='llm_call_limit', default=100,
                        help='openai call limit (default: 100)', required=False)
    parser.add_argument('--language', '-g', dest='language', default="english",
                        help='language for paper overview', required=False)
    parser.add_argument('--from-file', '-f', dest='from_file', default="", help="from chatbot dict file",
                        required=False)
    args = parser.parse_args()
    arxiv_paper_id = args.arxiv_paper_id
    openai_api_key = args.openai_api_key
    llm_call_limit = args.llm_call_limit
    language = args.language
    from_file = args.from_file

    print('downloading paper')
    paper = load_paper(arxiv_paper_id)
    print('downloading paper done')

    def _load_chatbot(openai_api_key: str, paper: Paper, language: str, from_file: str,
                      chat_handler: OpenAIChatHandler, progress_queue: Queue, done_queue: Queue):
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, callbacks=[chat_handler])

        embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

        if from_file:
            with open(from_file, 'r') as f:
                json_inputs = json.loads("\n".join(f.readlines()))
                chatbot = PaperChatbot.load(**json_inputs)
        else:
            chatbot = PaperChatbot.load(llm=llm, progress_queue=progress_queue,
                                        knowledge_vector_store=vectorstore,
                                        paper=paper, language=language)
            time.sleep(0.5)  # interval to resolve progress queue message
        done_queue.put(chatbot)

    # run background
    chat_handler = OpenAIChatHandler(call_count_limit=llm_call_limit)
    chatbot: Optional[PaperChatbot] = None
    chatbot_progress_queue = Queue(maxsize=0)
    done_queue = Queue(maxsize=0)
    threading.Thread(target=_load_chatbot,
                     kwargs={"openai_api_key": openai_api_key,
                             "paper": paper,
                             "language": language,
                             "from_file": from_file,
                             "chat_handler": chat_handler,
                             "progress_queue": chatbot_progress_queue,
                             "done_queue": done_queue}).start()
    for which, data in select(chatbot_progress_queue, done_queue):
        if which is chatbot_progress_queue:
            print(data)
        elif which is done_queue:
            chatbot = data
            print('chatbot loading done')
            break

    if not chatbot:
        return

    print("====== Paper Overview ======")
    print(chatbot.overview())
    print("============================")
    _interact_with_chatbot(chatbot=chatbot, language=language)


def _interact_with_chatbot(chatbot: PaperChatbot, language: str):
    chat_history: BaseChatMessageHistory = ChatMessageHistory()
    while True:
        try:
            user_input = ast.literal_eval('"' + input("Enter a command (exit/save/ask): ") + '"')
            if user_input.lower() == 'exit':
                print("Exiting the program.")
                break  # Exit the loop if the user enters 'exit'
            parts = user_input.split()
            if not parts:
                continue
            action = parts[0]
            if len(parts) >= 2:
                argument = ' '.join(parts[1:])
            else:
                argument = ""
            if action == "save":
                file_path = argument
                print(f"Contents saving to {file_path}")
                chatbot.save(file_path)
                print(f"Contents saved to {file_path}")
            elif action == "ask":
                def _get_answer(language: str, user_query: str, chat_history: BaseChatMessageHistory,
                                progress_queue: Queue, done_queue: Queue):
                    response = chatbot.answer(language=language, query=user_query, chat_history=chat_history,
                                              progress_queue=progress_queue)
                    done_queue.put(response)

                user_query = argument
                progress_queue = Queue(maxsize=0)
                done_queue = Queue(maxsize=0)
                threading.Thread(target=_get_answer,
                                 kwargs={"language": language,
                                         "user_query": user_query,
                                         "chat_history": chat_history,
                                         "progress_queue": progress_queue,
                                         "done_queue": done_queue}).start()
                for which, data in select(progress_queue, done_queue):
                    if which is progress_queue:
                        print(data)
                    elif which is done_queue:
                        chat_history.add_message(HumanMessage(content=user_query))
                        chat_history.add_message(AIMessage(content=data))
                        print(f"ANSWER: {data}")
                        break
            else:
                print("Invalid command. Supported actions are 'save' and 'ask' or 'exit'")
        except Exception as e:
            print(f"Unexpected error occurred. {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
