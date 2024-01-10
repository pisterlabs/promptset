import os
import json
import logging
from colorama import Fore, Style

from config import config
from documents import Docs
config()

# 初始化 pinecone
import pinecone
pinecone.init(
    api_key = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENVIRONMENT")
)

# 初始化bm25_encoder
from bm25 import load_bm25
bm25_encoder = load_bm25("./bm25.json")

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Pinecone
from langchain.retrievers import PineconeHybridSearchRetriever

# 构建和查询Pinecone数据索引的类

class Index:
    index_name = "langchain-pinecone-hybrid-search"
    store_file = "./backup/pinecone.json"
    if not os.path.exists(store_file):
        with open(store_file, "w") as f:
            json.dump({}, f)

    llm = ChatOpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()

    docsearch = Pinecone.from_existing_index(
            index_name,
            embeddings
            )
    if bm25_encoder:
        # use hybrid model
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=pinecone.Index(index_name),
            top_k=3
            )
    else:
        # use simple model
        retriever = docsearch.as_retriever(
            search_kwargs={"k": 3}
        )

    def query(input_text: str) -> list:
        """
        description: 查询远程索引
        :param input_text: str
        return: list[Document]
        """
        if bm25_encoder:
            result = Index.retriever.get_relevant_documents(input_text)
        else:
            result = Index.docsearch.similarity_search(
                input_text,
                include_metadata=True
                )
        text = "\n-----------\n".join([t.page_content for t in result])
        logging.info(f"""
pinecone search for '{input_text}':
There are {len(result)} results.
The most relevant results are:
{text}
            """)
        return result

    def upload_file(file_path: str) -> None:
        """
        description: 上传文件到pinecone
        :param file_path: str
        """
        with open(Index.store_file, "r", encoding="utf-8") as f:
            store = json.load(f)
        if file_path in store:
            print(Fore.BLUE + f"{file_path} has been uploaded" + Style.RESET_ALL)
            return

        # add file to pinecone
        docs = Docs.get_docs_from(file_path)
        for t in docs:
            try:
                Index.retriever.add_texts(t.page_content)
            except:
                logging.error(f"add_text error: \n{t.page_content[:100]}")

        # save file name
        store[file_path] = True
        print(Fore.BLUE + f"upload {file_path} to pinecone" + Style.RESET_ALL)
        with open(Index.store_file, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=4)

    def upload_files(directory: str, file_type: str="pdf") -> None:
        """
        description: 上传文件夹下的所有文件到pinecone
        :param directory: str
        :param file_type: str
        """
        files = Docs.get_files_from_directory(directory, file_type)
        for file in files:
            Index.upload_file(file)

if __name__ == "__main__":
    print(Fore.GREEN + "pinecone init:" ,pinecone.whoami() + Style.RESET_ALL)

    # 初始化远程索引
    Index.upload_files("./paper","pdf")

    ## create the index
    # pinecone.create_index(
    #     index_name,
    #     dimension = 1536,  # dimensionality of dense model
    #     metric = "dotproduct",  # sparse values supported only for dotproduct
    #     pod_type = "Starer",
    #     metadata_config = {"indexed": []}  # see explaination above
    # )