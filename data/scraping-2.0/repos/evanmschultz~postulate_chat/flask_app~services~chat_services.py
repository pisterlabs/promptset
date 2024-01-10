# TODO: Remove this file?

# from flask import flash
# from flask_app.services.vector_database import VectorDatabase

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import (
#     CharacterTextSplitter,
#     RecursiveCharacterTextSplitter,
# )
# from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.document_loaders import TextLoader
# from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI

# from flask_app.models.chat import Chat
# from flask_app import database

# # from langchain.document_loaders import TextLoader
# # from langchain.document_loaders import PyPDFLoader


# class ChatService:
#     def __init__(self, user_id: int, client=None, chat_id: int = 0):
#         self.chat = self.get_or_create_chat(chat_id)
#         # self.chat_history: list = []
#         # self.db_query: str = ""
#         # self.db_response: list = []
#         # # self.vector_db: VectorDatabase = VectorDatabase()
#         # if not chat_id:
#         #     self.chat: Chat = Chat()
#         # self.memory: ConversationBufferMemory = ConversationBufferMemory(
#         #     memory_key="chat_history", return_messages=True
#         # )

#     def get_or_create_chat(self, user_id, chat_id: int = 0):
#         if chat_id:
#             chat: Chat = Chat.query.get(id=chat_id, user_id=user_id)
#             if chat is None:
#                 flash("Chat not found", "error")
#                 return None  # or you could raise an exception
#             return chat
#         else:
#             new_chat = Chat(user_id=user_id)
#             database.session.add(new_chat)
#             database.session.commit()
#             return new_chat

#     # def convchain(self, query):
#     #     if not query:
#     #         return ""
#     # result = self.vector_db.query(
#     #     {"question": query, "chat_history": self.chat_history}
#     # )
#     # self.chat_history.extend([(query, result["answer"])])
#     # self.db_query = result["generated_question"]
#     # self.db_response = result["source_documents"]
#     # return result["answer"]
