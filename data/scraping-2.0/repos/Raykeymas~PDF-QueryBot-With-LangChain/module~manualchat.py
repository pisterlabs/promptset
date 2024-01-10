from typing import List
from .vectorstore import VectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


class ManualChat:
  def __init__(self, vectorstore: VectorStore, model_name: str) -> None:
    """
    ManualChatオブジェクトを初期化します。

    Args:
        vectorstore (VectorStore): VectorStoreオブジェクト

    """
    self.llm = ChatOpenAI(temperature=0, model_name=model_name)
    self.qa = ConversationalRetrievalChain.from_llm(self.llm, vectorstore.as_retriever(), return_source_documents=True)
    self.chat_history: List[str] = []

  def ask(self, question: str) -> dict:
    """
    チャットボットに質問をし、応答を返します。

    Args:
        question (str): チャットボットに尋ねる質問

    Returns:
        dict: チャットボットの応答を含むdict
    """
    result = self.qa({"question": question, "chat_history": self.chat_history})
    self.chat_history.append((question, result["answer"]))
    return result
