import os 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import AstraDB
from astrapy.db import AstraDBCollection
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document

ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"] 
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

user_collection = AstraDBCollection(
        collection_name="user_documents",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )

vstore = AstraDB(
        embedding=embeddings,
        collection_name="astra_vector_demo",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
welcome_message = """Welcome to the Build your own Custom GPT demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


def process_file(file: AskFileResponse):
    app_user = cl.user_session.get("user")
    import tempfile    
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tempfile:
        if file.type == "text/plain":
            tempfile.write(file.content)
        elif file.type == "application/pdf":
            with open(tempfile.name, "wb") as f:
                f.write(file.content)

        loader = Loader(tempfile.name)
        docs = loader.load_and_split(text_splitter=text_splitter)        
        for doc in docs:
            doc.metadata["source"] = f"{file.name}"
            doc.metadata["username"] = f"{app_user.username}"
        return docs

def get_docsearch(file: AskFileResponse):
    docs = process_file(file)    
    cl.user_session.set("docs", docs)
    user = cl.user_session.get("dbuser")    
    vstore.add_documents(docs)    
    user["files"].append(f"{file.name}")
    user_collection.update_one(filter={"username": f"{user['username']}"}, update={"$set": {"files": user['files']}}) 
    return vstore

def get_files_for_user(user):
    collection = AstraDBCollection(
        collection_name="user_documents",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
    user = collection.find_one({"username": f"{user.username}"})
    cl.user_session.set("dbuser", user["data"]["document"])
    return user["data"]["document"]

async def upload_new_file():
    app_user = cl.user_session.get("user")    
    files = await cl.AskFileMessage(
                    content=welcome_message,
                    accept=["text/plain", "application/pdf"],
                    max_size_mb=20,
                    timeout=180,
                    disable_human_feedback=True,
    ).send()
    file = files[0]
    msg = cl.Message(
            content=f"Processing `{file.name}`...", 
            disable_human_feedback=True
    )
    await msg.send()  
    dbuser = cl.user_session.get('dbuser')
    if not dbuser: 
        newuser = {"username": f"{app_user.username}", 
                   "files": [f"{file.name}"]}
        user_collection.insert_one(newuser)
        user=user_collection.find_one({"username": f"{app_user.username}"})
        cl.user_session.set("dbuser", user["data"]["document"])
    await cl.make_async(get_docsearch)(file)
    msg.content = f"Processing done. You can now ask questions!"
    await msg.update()
