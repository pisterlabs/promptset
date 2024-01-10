from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from chatglm_llm import ChatGLM
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.document_loaders import PyPDFLoader, CSVLoader
import requests
import json

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "../text2vec-large-chinese"
}

llm_model_dict = {
    "chatglm-6b": "THUDM/chatglm-6b",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4"
}


class CustomGLM(LLM):

    url: str
    temperature: float = 0.01
    max_length: int = 4096
    top_p: float = 0.7
    history: list = []

    @property
    def _llm_type(self) -> str:
        return "ChatGLM-client"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        data = {"prompt": prompt,
                "temperature": self.temperature,
                "max_length": self.max_length,
                "history": self.history,
                "top_p": self.top_p}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        return response['response']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"url": self.url}
    

chatglm = CustomGLM(url="http://10.102.32.68:9101/")
# chatglm = ChatGLM()
# chatglm.load_model(model_name_or_path=llm_model_dict["chatglm-6b"])

def init_knowledge_vector_store(filepath):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict["text2vec"], )
    # loader = UnstructuredFileLoader(filepath, mode="elements")
    # loader = CSVLoader(file_path='../datasets_test.csv')
    loader = CSVLoader(file_path='../处治方案知识库0.csv')
    docs = loader.load()

    # vector_store = Chroma.from_documents(docs, embeddings, persist_directory='./store')
    vector_store = Chroma.from_documents(docs, embeddings)
    return vector_store


def get_knowledge_based_answer(query, vector_store, chat_history=[]):
    system_template = """基于以下内容，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "不知道" 或 "没有足够的相关信息"，不要试图编造答案。答案请使用中文。
    ----------------
    {context}
    ----------------
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    condese_propmt_template = """任务: 给一段对话和一个后续问题，将后续问题改写成一个独立的问题。确保问题是完整的，没有模糊的指代。
    ----------------
    聊天记录：
    {chat_history}
    ----------------
    后续问题：{question}
    ----------------
    改写后的独立、完整的问题："""
    new_question_prompt = PromptTemplate.from_template(condese_propmt_template)
    chatglm.history = chat_history
    knowledge_chain = ConversationalRetrievalChain.from_llm(
        llm=chatglm,
        retriever=vector_store.as_retriever(),
        qa_prompt=prompt,
        condense_question_prompt=new_question_prompt,
    )

    knowledge_chain.return_source_documents = True
    # knowledge_chain.top_k_docs_for_context = 2

    result = knowledge_chain({"question": query, "chat_history": chat_history})
    return result, chatglm.history


if __name__ == "__main__":
    # filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
    filepath = "./"
    vector_store = init_knowledge_vector_store(filepath)
    history = []
    # while True:
    # query = input("Input your question 请输入问题：")
    query = "大路梁子隧道的灾害发生时间"
    resp, history = get_knowledge_based_answer(query=query,
                                            vector_store=vector_store,
                                            chat_history=history)
    print("回答回答：", resp['answer'])
    # print("历史历史：", history)
    # query = "那这条隧道的灾害情况是什么"
    # resp, history = get_knowledge_based_answer(query=query,
    #                                         vector_store=vector_store,
    #                                         chat_history=history)
    # print("回答回答：", resp['answer'])