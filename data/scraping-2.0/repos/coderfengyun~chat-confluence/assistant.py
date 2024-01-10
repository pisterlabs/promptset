
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import (
    Tool,
    AgentOutputParser,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.prompts import BaseChatPromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
import re
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

def initiate_agent():
    embeddings = OpenAIEmbeddings(deployment="embeddings_model_jp", chunk_size=1, openai_api_version="2023-03-15-preview")
    redis_url_str = os.environ["REDIS_URL"]
    db = Redis.from_existing_index(embeddings, redis_url=redis_url_str, index_name='chat_confluence_new')
    retriever = db.as_retriever()
    llm = AzureChatOpenAI(deployment_name="chat-confluence-jp-0613", model_name="gpt-35-turbo-16k", verbose=True, openai_api_version="2023-03-15-preview")
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)
    qa = CustomConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=False, memory=memory, verbose=True)
    return qa

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain, _get_chat_history
from typing import Dict, Any, Optional, List
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import Document

class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    last_source_documents: List[Document] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_source_documents = None
        
    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs['question']
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs['chat_history'])
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = self.question_generator.run(question=question, chat_history=chat_history_str, callbacks=callbacks)
        else:
            new_question = question
        docs = self._get_docs(new_question, inputs)
        new_inputs = inputs.copy()
        new_inputs['question'] = new_question
        new_inputs['chat_history'] = chat_history_str
        answer = self.combine_docs_chain.run(input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs)
        self.last_source_documents = docs
        return {self.output_key: answer}