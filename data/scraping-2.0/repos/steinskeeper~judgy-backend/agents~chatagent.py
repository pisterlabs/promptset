from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from bson import ObjectId
from fastapi import APIRouter, Request
from langchain.document_loaders import GitLoader
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatVertexAI
import os
from langchain.memory import VectorStoreRetrieverMemory
from pymongo import MongoClient
load_dotenv()


router = APIRouter()
client = MongoClient("mongodb://localhost:27017/")
db = client["judgy"]


@router.get("/chat-agent")
def chatAgent_endpoint():
    return {"message": "Hello from Chat Agent"}


@router.post("/chat-agent")
async def invoke_chat_agent(request: Request):

    data = await request.json()
    technologies = ""
    theme = ""
    isAllowed = False
    for x in db.hackathons.find():
        technologies = x["technologies"]
        theme = x["theme"]
        if "isAllowed" in x:
            isAllowed = x["isAllowed"]
        else:
            isAllowed = False
        break
    if isAllowed == False:
        return {"answer": "Sorry, We have reached our credit limit.", "chathistory": []}
    print("Data",data)
    project = db.projects.find_one({"_id": ObjectId(data["project_id"])})
    
    DIRECTORY = "projects_source_code/"+data["project_id"]
    loader = DirectoryLoader(DIRECTORY, silent_errors=True)
    print("loader", loader)
    llm = ChatVertexAI()
    index = VectorstoreIndexCreator().from_loaders([loader])
    print("index creation", index)
    retriever = index.vectorstore.as_retriever()
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    print("before context memory", memory)
    memory.save_context(
        {"input": "Idea : "+project["shortDescription"]}, {"output": "..."})
    memory.save_context(
        {"input": "Theme for the Hackathon : "+theme}, {"output": "..."})
    memory.save_context(
        {"input": "Technologies that must be used for the hackathon project : "+technologies}, {"output": "..."})
    
    print("after context memory", memory)

    _DEFAULT_TEMPLATE = """The following is a conversation between a hackathon judge and an AI. 
    The AI is a market researcher and a code reviewer. 
    Rules for answering: 
    1. Use statistical data where ever possible.
    2. Remember to answer like a market researcher.
    3. Answer the question as best you can, in a paragraph.
    4. You must answer in one paragraph. Do not use formatting.
    5. Your paragraph must not have more than 70 words.
    6. You must analyze all the files in the project when a code related question is asked.
    Relevant pieces of previous conversation:
    {history}

    (You do not need to use these pieces of information if not relevant)

    Current conversation:
    Human: {input}
    AI:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
    )
    conversation_with_summary = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=memory,
        verbose=True
    )
    aiResp = conversation_with_summary.predict(input=data["question"])
    chatHistory = data["chathistory"]
    chatHistory.append(
        {"input": data["question"], "output": aiResp})
    
    return {
        "answer": aiResp,
        "chathistory": chatHistory
    } 
