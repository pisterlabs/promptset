import hashlib
import os
import time
from bs4 import BeautifulSoup as Soup
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from sky_langchain.loaders.recursive_url_loader import RecursiveUrlLoader
from sky_langchain.models.chatgpt import chatgpt_three_point_five_turbo
from sky_langchain.wrapper import ChatChainWrapper
from bs4 import BeautifulSoup as Soup


def pdf_reader(
    name="guofulun",
    pdf_path="/Users/zhangsikai/Downloads/guofulun.pdf",
):
    cache_dir = os.path.join(".embeddings/", name)
    llm = chatgpt_three_point_five_turbo
    embedding = OpenAIEmbeddings()
    if not os.path.exists(cache_dir):
        loader = PyPDFLoader(pdf_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = loader.load_and_split(splitter)
        vstore = Chroma(
            collection_name="my_collection",
            embedding_function=embedding,
            persist_directory=cache_dir,
        )
        batch_size = 200
        total = len(splits)
        progress = 0
        for split in range(0, len(splits), batch_size):
            vstore.add_documents(splits[split : split + batch_size])
            progress += batch_size
            print(f"{progress}/{total}")
            time.sleep(30)
    else:
        vstore = Chroma(
            collection_name="my_collection",
            embedding_function=embedding,
            persist_directory=cache_dir,
        )

    retriever = vstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    wrapper = ChatChainWrapper(llm, retriever, chinese=True)
    wrapper.run()


pdf_reader()
