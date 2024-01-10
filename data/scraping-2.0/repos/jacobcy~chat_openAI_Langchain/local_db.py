import os
import json
import logging
from colorama import Fore, Style

from config import config
from documents import Docs
config()

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 使用Chroma本地处理数据
class LocalDB:
    embeddings = OpenAIEmbeddings()
    store_file = "./backup/chroma.json"
    if not os.path.exists(store_file):
        with open(store_file, "w") as f:
            json.dump({}, f)

    def query(input_text: str, directory: str="db") -> list:
        """
        description: 查询本地索引
        :param input_text: str
        :param directory: str
        return: list[Document]
        """
        # Now we can load the persisted database from disk
        local_db = Chroma(
            embedding_function=LocalDB.embeddings,
            persist_directory=directory
            )

        # search for similar documents
        # result = local_db.similarity_search(input_text)
        # local_db = None

        # search for retrieval documents
        retriever = local_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
            )
        result = retriever.get_relevant_documents(input_text)

        text = "\n-----------\n".join([t.page_content for t in result])
        logging.info(f"Chroma search for '{input_text}':\n{text}")
        return result

    def save(docs: list, directory: str="db") -> None:
        """
        description: 数据持久化
        :param docs: list[Document]
        :param directory: str
        """
        local_db = Chroma.from_documents(
            documents=docs,
            embedding=LocalDB.embeddings,
            persist_directory=directory
            )
        local_db.persist()
        local_db = None

    def upload_file(file_path: str) -> bool:
        """
        description: 上传文件到本地索引
        :param file_path: str
        """
        with open(LocalDB.store_file, "r", encoding="utf-8") as f:
            store = json.load(f)
        if file_path in store:
            print(Fore.BLUE + f"{file_path} has been uploaded" + Style.RESET_ALL)
            return True

        # add file to chroma
        docs = Docs.get_docs_from(file_path)
        try:
            LocalDB.save(docs)
        except:
            logging.error(f"save {file_path} to Chroma failed")
            return False
        
        # save file name
        store[file_path] = True
        print(Fore.GREEN + f"upload {file_path} to Chroma" + Style.RESET_ALL)
        with open(LocalDB.store_file, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=4)
        return True

    def upload_files(directory: str, file_type: str="pdf") -> None:
        """
        description: 上传文件夹下的所有文件到本地索引
        :param directory: str
        :param file_type: str
        """
        files = Docs.get_files_from_directory(directory, file_type)
        for file in files:
            LocalDB.upload_file(file)

if __name__ == "__main__":
    # 初始化本地索引
    LocalDB.upload_files("./paper", "docx")
