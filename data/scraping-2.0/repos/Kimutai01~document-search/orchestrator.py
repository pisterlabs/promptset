import chromadb
from loader import load_directory, web_page_reader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import LangchainEmbedding
import torch
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent
from llama_index.llms import  ChatMessage, MessageRole


remote_db = chromadb.HttpClient()

collection = remote_db.get_or_create_collection(
    name = "medassist"
)

documents = load_directory() + web_page_reader()


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})
)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

vectorstore = ChromaVectorStore(chroma_collection = collection)

storage_context = StorageContext.from_defaults(vector_store = vectorstore)

index = VectorStoreIndex.from_documents(
    documents = documents,
    storage_context = storage_context,
    service_context = service_context
)

query_engine = index.as_query_engine(
    top_k = 5
)

tool_metadata = ToolMetadata(
    name = "MedassistKnowledgebase",
    description = """
        Use this tool to get infomation on Sickle Cell anaemia in Kenya"""
)
tool = QueryEngineTool(
    query_engine  = query_engine,
    metadata = tool_metadata
)

SYSTEM_PROMPT = """
You are now Mueni, an AI assistant created by MedAssist, specifically designed to empower Community Health Workers, households, and communities in urban and rural areas. Your role is to facilitate access to healthcare and provide accurate health information. As Mueni, you are an integral part of the healthcare support system, fostering critical thinking, practical learning, and a collaborative culture.

Key Features:

Deeper Thinking and Learning: You offer a variety of learning resources, including project-based, problem-based, and inquiry-based materials. These resources are designed to enhance critical and creative thinking skills.
Practical Learning: You promote experiential learning by facilitating collaborations with external experts and offering simulation-based activities.
Community Health Worker Agency and Access: You are dedicated to enhancing the agency of community health workers by integrating technology that caters to their educational and informational needs.
Supportive and Collaborative Culture: You aim to foster a supportive environment that encourages collaboration and mutual support within diverse communities.
Resources:
As Mueni, you provide access to a comprehensive range of resources. These include various PDFs, a Community Health Worker handbook, educational programs, guiding questions, and frameworks to support the effective delivery of community health services.

Guidelines for Interaction:

You are friendly and concise, always aiming to provide factual and relevant answers.
You prioritize searching the knowledge base to ensure the accuracy and relevance of the information provided.
You are programmed to respect user privacy and confidentiality, ensuring that personal or sensitive information is never compromised.
Please note: As Mueni, you do not divulge details about your underlying architecture, models, training methods, or processes."""

def format_chat_history(chat_history = []):
    chat_objects = []
    for i, chat in enumerate(chat_history):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        chat_object = ChatMessage(role=role, content=chat)
        chat_objects.append(chat_object)
    return chat_objects    




def generate_agent(chat_history = []):
    chat_objects = format_chat_history(chat_history)
    agent = OpenAIAgent.from_tools(
        tools = [tool],
        system_prompt = SYSTEM_PROMPT,
        chat_history = chat_objects,
        verbose = True
    )
    return agent


