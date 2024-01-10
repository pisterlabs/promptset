import os
import pathlib
from datetime import datetime
from typing import Optional, Type

from config import apikeys

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun, get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain import HuggingFaceHub, OpenAI, PromptTemplate
from langchain.vectorstores import Chroma

os.environ['OPENAI_API_KEY'] = apikeys.OPENAI_API_KEY


from typing import Optional

class EmbeddingProvider(BaseTool):
    name = "embedding_provider"
    description = "Useful when you need to get an answer to only questions regarding Koelnmesse, Trade Fairs of the Koelnmesse or Koelnmesse specific links from a user. Never ask questions unrelated to the Koelnmesse or its Trade Fairs."
    llm = ChatOpenAI(temperature=0)
    embedding = OpenAIEmbeddings()
    chat_room: Optional[str] = None
    chat_message: str = ''
    keys_to_retrieve: int = 8

    def __init__(self, chat_room=None, chat_message='', keys_to_retrieve=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_room = chat_room
        self.chat_message = chat_message
        self.keys_to_retrieve = keys_to_retrieve

    def format_memory(self, messages) -> str:
        memory = []
        format_template = """["type": {message_type}, "content": {content}]"""

        template = PromptTemplate(
            template=format_template,
            input_variables=['message_type', 'content']
        )

        for message in messages:
            memory_input = template.format(message_type=message.type, content=message.content)
            memory.append(memory_input)

        return str(memory)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        vectordb = Chroma(
                persist_directory=f"training/vectorstores/",
                embedding_function=self.embedding)

        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": self.keys_to_retrieve})

        # create a chain to answer questions
        qa = ConversationalRetrievalChain.from_llm(self.llm, retriever, chain_type='stuff',
                                                   return_source_documents=True)

        chat_history = []

        temp_message = ''

        for message in self.chat_room['chat']:
            if message['type'] == 'User':
                temp_message = message['content']
            else:
                chat_history.append((temp_message, message['content']))

        print(chat_history)
        print(self.keys_to_retrieve)

        vectordbkwargs = {"search_distance": 0.9}
        with get_openai_callback() as cb:
            result = qa({"question": self.chat_message, "chat_history": chat_history, "vectordbkwargs": vectordbkwargs})
            print(cb)

        print(result['source_documents'])

        return result['answer']

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("star_coder does not support async")