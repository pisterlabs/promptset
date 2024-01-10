from typing import List

import langchain
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.tools import BaseTool

langchain.verbose = True


# Index作成
def create_index() -> VectorStoreIndexWrapper:
    '''
    loader = PyPDFLoader("./src/pdzac.pdf") # Vectorソースの指定
    loader = DirectoryLoader("./src/JP2_html_201020_2/", glob="**/*.html", loader_cls=UnstructuredHTMLLoader) # フォルダ内のHTMLをすべて学習させる場合
    '''
    loader = DirectoryLoader("./src/", glob="**/*.txt") # フォルダ内のテキストをすべて学習させる場合
    return VectorstoreIndexCreator().from_loaders([loader])


def chat(
    message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper
) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return index.query(question=message, llm=llm)