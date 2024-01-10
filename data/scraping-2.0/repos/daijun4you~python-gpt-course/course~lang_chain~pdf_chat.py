from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# 记得api_key改成你的
openai_api_key = "sk-xxxx"


def run():
    # 读取pdf文件
    pdf_reader = PdfReader(os.path.dirname(
        os.path.abspath(__file__)) + "/pdf_chat_demo.pdf")
    pdf_content = ""
    for page in pdf_reader.pages:
        pdf_content += page.extract_text()

    # 将pdf内容进行分片
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
    )
    split_docs = text_splitter.split_text(pdf_content)

    # 利用OpenAI进行Embedding并存储到向量数据库中，
    docsearch = Chroma.from_texts(split_docs, OpenAIEmbeddings(
        openai_api_key=openai_api_key
    ))

    # 创建问答Chain
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo-16k-0613"
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(), return_source_documents=False,
    )

    result = qa({"query": "什么瓜不能吃"})

    print(result)


if __name__ == "__main__":
    run()
