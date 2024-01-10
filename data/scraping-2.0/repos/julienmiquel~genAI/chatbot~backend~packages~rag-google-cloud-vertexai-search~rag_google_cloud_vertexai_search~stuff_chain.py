import os

from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.retrievers import GoogleVertexAISearchRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA

# Get project, data store, and model type from env variables
PROJECT_ID  = os.environ.get("GCP_PROJECT_ID")
REGION      =  os.environ.get("GCP_REGION")

DATA_STORE_ID           = os.environ.get("DATA_STORE_ID")
DATA_STORE_LOCATION_ID  =  os.environ.get("DATA_STORE_LOCATION_ID")
DATA_STORE_MAX_DOC = os.environ.get("DATA_STORE_MAX_DOC", 3)

LLM_CHAT_MODEL_VERSION  = os.environ.get("LLM_CHAT_MODEL_VERSION")
LLM_TEXT_MODEL_VERSION  = os.environ.get("LLM_TEXT_MODEL_VERSION")


if not DATA_STORE_ID:
    raise ValueError(
        "No value provided in env variable 'DATA_STORE_ID'. "
        "A  data store is required to run this application."
    )
os.environ["DATA_STORE_ID"] = DATA_STORE_ID
os.environ["PROJECT_ID"] = PROJECT_ID
os.environ["LOCATION_ID"] = DATA_STORE_LOCATION_ID
os.environ["REGION"] = REGION
os.environ["MODEL"] = LLM_TEXT_MODEL_VERSION


llm = VertexAI(model_name=LLM_TEXT_MODEL_VERSION)


# Create Vertex AI retriever
retriever = GoogleVertexAISearchRetriever(
    project_id=PROJECT_ID, 
    data_store_id=DATA_STORE_ID, 
    location_id = DATA_STORE_LOCATION_ID,
    max_documents=DATA_STORE_MAX_DOC,
    engine_data_type=1, # structured data, 
    get_extractive_answers = True
)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

#retrieval_qa_with_sources({"question": search_query}, return_only_outputs=True)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversational_retrieval = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory
)

from langchain.prompts import PromptTemplate

prompt_template = """Use the context to answer the question at the end.
You must always use the context and context only to answer the question. Never try to make up an answer. If the context is empty or you do not know the answer, just say "Je suis désolé, je ne connais pas cette information.".
The answer is always few sentences long and in French.

Context: {context}

Question: {question}
Helpful Answer in French:
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
qa_chain = RetrievalQA.from_llm(
    llm=llm, prompt=prompt, retriever=retriever, return_source_documents=True
)

#-------------


# Add typing for input
class Question(BaseModel):
    __root__: str

retrieval_qa = retrieval_qa.with_types(input_type=Question)
qa_chain = qa_chain.with_types(input_type=Question)
conversational_retrieval= conversational_retrieval.with_types(input_type=Question)