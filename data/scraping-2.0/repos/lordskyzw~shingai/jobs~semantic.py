from pymongo import MongoClient
import os
from langchain.memory import MongoDBChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


##########################       STEP ONE          ####################################
########### Retrieve recipient phone numbers from MongoDB and add them to a list
client = MongoClient(
    "mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200"
)
database = client["users"]
collection = database["recipients"]
# first make a set to avoid getting identical phone numbers:
phone_numbers = set()
recipients = collection.find()
for recipient in recipients:
    phone_number = recipient["phone_number"]
    phone_numbers.add(phone_number)
# convert back to list
phone_numbers = list(phone_numbers)

########################        STEP TWO            ################################
# setting up the vectorstore
embeddings = OpenAIEmbeddings()  # type: ignore
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),  # type: ignore
    environment="northamerica-northeast1-gcp",
)
index = pinecone.Index(index_name="thematrix")
vectorstore = Pinecone(index, embeddings.embed_query, "text")


# upsert everyones vector embeddings
for phone_number in phone_numbers:
    history = MongoDBChatMessageHistory(
        connection_string="mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200",
        session_id=phone_number,
        database_name="test",
        collection_name="message_store",
    )
    entire_history = str(history.messages).replace(
        ", additional_kwargs={}, example=False", ""
    )
    entire_history.replace("content=", "")

    char_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    doc_texts = char_text_splitter.split_text(entire_history)
    vectorstore.add_texts(texts=doc_texts, namespace=phone_number)
    print(f"added semantic memories for: {phone_number}")
