import os

import openai
import qdrant_client
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Qdrant
from pymongo.database import Database

from ..utils.security import hash_password

load_dotenv()


class ChatRepository:
    def __init__(self, database: Database):
        self.database = database

    def get_response(self, user_question: str) -> str:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        client = qdrant_client.QdrantClient(
            os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY")
        )

        embeddings = OpenAIEmbeddings()
        # db = Chroma(embedding_function=embeddings, persist_directory="app/database")
        # db.persist()
        vector_store = Qdrant(
            client=client,
            collection_name="civil-main",
            embeddings=embeddings,
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.0)
        chain = load_qa_chain(llm=llm, chain_type="map_reduce")
        docs = vector_store.similarity_search(user_question)
        # qa = RetrievalQA.from_chain_type(
        #     llm=llm, chain_type="map_reduce", retriever=vector_store.as_retriever()
        # )
        return chain.run({"input_documents": docs, "question": user_question})
        # return qa.run(user_question)
        # payload = {
        #     "email": user["email"],
        #     "password": hash_password(user["password"]),
        #     "created_at": datetime.utcnow(),
        # }

        # self.database["message"].insert_one(payload)

    # def get_user_by_id(self, user_id: str) -> Optional[dict]:
    #     user = self.database["users"].find_one(
    #         {
    #             "_id": ObjectId(user_id),
    #         }
    #     )
    #     return user

    # def get_user_by_email(self, email: str) -> Optional[dict]:
    #     user = self.database["users"].find_one(
    #         {
    #             "email": email,
    #         }
    #     )
    #     return user
