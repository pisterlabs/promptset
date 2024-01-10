import os
import time

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

pdf_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'test_data', 'pdfs'))
persist_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'chroma_persist'))
project_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

load_dotenv(dotenv_path=os.path.join(project_dir, '.env'))


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result

    return wrapper


@time_it
def main():
    print(pdf_dir)
    # Here set need_load to True to load the pdf files and create the Chroma vector database
    need_load = False
    embeddings = OpenAIEmbeddings()
    if need_load:
        # 加载 pdf 文件
        loader = PyPDFDirectoryLoader(pdf_dir)
        documents = loader.load()
        # 初始化加载器
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        # 切割加载的 document
        split_docs = text_splitter.split_documents(documents)

        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
        # 持久化 Chroma 向量数据库
        docsearch.persist()

    # 加载数据
    docsearch = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = docsearch.as_retriever()

    system_template = """
    Use the following context to answer the user's question.
    If you don't know the answer, say you don't, don't try to make it up.
    -----------

    -----------
    {chat_history}
    """

    # 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{question}')
    ]

    # 初始化 prompt 对象
    prompt = ChatPromptTemplate.from_messages(messages)

    # 初始化问答链
    history = ConversationBufferMemory()
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, max_tokens=2048), retriever,
                                               condense_question_prompt=prompt)

    chat_history = []

    while True:
        question = input('Question：')
        # 开始发送问题 chat_history 为必须参数,用于存储对话历史
        result = qa({'question': question, 'chat_history': chat_history})
        chat_history.append((question, result['answer']))
        print(result['answer'])


if __name__ == '__main__':
    main()
