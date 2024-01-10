from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import AzureChatOpenAI
from langchain.experimental import AutoGPT
from langchain.memory import PostgresChatMessageHistory
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

# from callback import CustomHandler
from dotenv import load_dotenv
import os
import faiss

# IMPORT TOOL START
# from tools.docsimport import docsimport
#from tools.zapiertool import zapiertool
from custom_tools import customtools
from python import pythontool
# IMPORT TOOL END

load_dotenv()

status = "Preparing"
memories = {}
history = {}
agents = {}
agent_chains = {}
tools = []

azchat=AzureChatOpenAI(
    client=None,
    openai_api_base=str(os.getenv("OPENAI_API_BASE")),
    openai_api_version="2023-03-15-preview",
    deployment_name=str(os.getenv("CHAT_DEPLOYMENT_NAME")),
    openai_api_key=str(os.getenv("OPENAI_API_KEY")),
    openai_api_type = "azure"
)

# ADD TOOL START
# tools.extend(docsimport(os.getenv("TOOLS_CATEGORY"), azchat))
#tools.extend(zapiertool())
tools.extend(customtools())
tools.extend(pythontool())
# ADD TOOL END

tool_names = [tool.name for tool in tools]

def SetupChatAgent(id):
    print("------TOOLS------")
    print(tool_names)
    postgresUser = str(os.getenv("POSTGRES_USER"))
    postgresPassword = str(os.getenv("POSTGRES_PASSWORD"))
    postgresHost = str(os.getenv("POSTGRES_HOST"))
    postgresPort = str(os.getenv("POSTGRES_PORT"))
    embeddings_model = OpenAIEmbeddings(client=None, model=str(os.getenv("EMBEDDING_DEPLOYMENT_NAME")), chunk_size=1)
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    memories[id] = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    history[id] = PostgresChatMessageHistory(
        connection_string=f"postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/chat_history", 
        session_id=str(id))
    agent_chains[id] = AutoGPT.from_llm_and_tools(
        ai_name="GitLab AI",
        ai_role="You are an GitLab AI assistant that helps people find information.",
        tools=tools,
        llm=azchat,
        memory=memories[id].as_retriever()
    )
    agent_chains[id].chain.verbose = True


class MessageReq(BaseModel):
    id: str
    text: str

class MessageRes(BaseModel):
    result: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def keepAsking(mid, text):
    res = ""
    try:
        res = agent_chains[mid].run(input=text)
    except:
        res = keepAsking(mid, text)
    return res

def clearMemory(mid):
    newMessage = str(os.getenv("CHAT_SYSTEM_PROMPT")) + "\nThe summary as below:\n" + agent_chains[mid].memory.predict_new_summary(agent_chains[mid].memory.buffer, agent_chains[mid].memory.moving_summary_buffer)
    agent_chains[mid].memory.buffer.clear()
    agent_chains[mid].memory.save_context({"input": newMessage}, {"ouputs": "I will try my best to help for the upcoming questions."})

@app.post("/run")
def run(msg: MessageReq):
    if (msg.id not in agent_chains):
        SetupChatAgent(msg.id)
    response = agent_chains[msg.id].run([msg.text])
    history[msg.id].add_user_message(msg.text)
    history[msg.id].add_ai_message(response)

    result = MessageRes(result=response)
    return result

@app.get("/tools")
def get_tools():
    tool_list = []
    for tool in tools:
        tool_dict = {"name": tool.name, "description": tool.description}
        tool_list.append(tool_dict)
    return {"tools": tool_list}

@app.get("/status")
def get_status():
    return {"status": "OK"}