import asyncio
import os
from typing import Any, Callable, Dict, List

import pinecone
from fastapi import UploadFile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from ..base import BaseTool
from .files import CustomFileProcessor


class FileProcessTool(BaseTool):
    name: str = "file_process"
    description: str = "Tool for process files to improve the discussion"
    user_description: str = "You can enable this to upload some files or submit a link for discussion. It is suggested that you remove any uploaded files after you are done, to have a better conversation flow."
    usable_by_bot = False

    def __init__(self, func: Callable=None, **kwargs):
        self.word_limit = kwargs.get("file_word_limit", 1000)
        self.top_k = kwargs.get("file_top_k", 2)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"] # for langchain compatibility
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.word_limit, chunk_overlap=int(self.word_limit / 5))

        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_API_ENV"])
        index_name = "fileidx"
        dimension = 1536
        # uncomment these for real production, slow check
        # try:
        #     if index_name not in pinecone.list_indexes():
        #         print("Creating pinecone index")
        #         pinecone.create_index(index_name, dimension)
        # except Exception as e:
        #     print(e)

        self.pinecone_db = Pinecone(index=pinecone.Index(index_name), embedding=self.embeddings, text_key="text")

        self.current_user_files: Dict[str, List[str]] = {} # this dictionary maps user id to a list of document ids recorded in the database
        OnStartUp = kwargs.get("OnStartUp")
        OnStartUpMsgEnd = kwargs.get("OnStartUpMsgEnd")
        OnUserMsgReceived = kwargs.get("OnUserMsgReceived")
        OnResponseEnd = kwargs.get("OnResponseEnd")
        OnUserDisconnected = kwargs.get("OnUserDisconnected")
        
        OnUserMsgReceived += self.OnUserMsgReceived
        OnUserDisconnected += self.OnUserDisconnected

        super().__init__(None)
    
    async def update_user_doc(self, user_id, current_file: CustomFileProcessor):
        splitted_doc = current_file.generate_docs()
        self.remove_user_files(user_id)
        self.current_user_files[user_id] = self.pinecone_db.add_documents(splitted_doc, namespace=f"{user_id}-files")
    
    async def handle_url(self, user_id, url):
        print(f"user id: {user_id}, url: {url}")
        try:
            current_file = CustomFileProcessor(file_url=url)
            await current_file.load_file()
            task = asyncio.create_task(self.update_user_doc(user_id, current_file))
            await task
            return {"user_id": user_id, "url": url, "status": "success"}
        except Exception as e:
            print(e)
            return {"user_id": user_id, "status": "failed", "detail": str(e)}
    
    async def handle_file_upload(self, user_id, file: UploadFile):
        try:
            print(f"Upload file: user id: {user_id}, file name: {file.filename}, file header: {file.headers}")
            current_file = CustomFileProcessor(file=file)
            await current_file.load_file()
            task = asyncio.create_task(self.update_user_doc(user_id, current_file))
            await task
            return {"user_id": user_id, "filename": file.filename, "status": "success"}
        except Exception as e:
            print(e)
            return {"user_id": user_id, "status": "failed", "detail": str(e)}
    
    def OnUserMsgReceived(self, **kwargs):
        user_assistants = kwargs.get("user_assistants", None)
        user_msg = kwargs.get("message")["content"]
        user_id = kwargs.get("user_id")
        if (user_assistants is None) or (user_id not in self.current_user_files):
            return
        
        docs = self.pinecone_db.as_retriever(search_kwargs={"k": self.top_k, "namespace": f"{user_id}-files"}).get_relevant_documents(user_msg)
        all_text = "\n\n".join([doc.page_content for doc in docs])
        print(len(docs))
        print(all_text)
        for user_assistant_update in user_assistants:
            user_assistant_update(all_text)

    def OnUserDisconnected(self, **kwargs):
        user_id = kwargs.get("user_id")
        self.remove_user_files(user_id)
    
    def remove_user_files(self, user_id):
        if user_id in self.current_user_files and self.current_user_files[user_id] is not None:
            self.pinecone_db.delete(ids=self.current_user_files[user_id], namespace=f"{user_id}-files") 
    
    def on_enable(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_disable(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _run(self):
        return None