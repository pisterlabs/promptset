import json
from typing import List, Tuple

from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.embeddings import GPT4AllEmbeddings, CacheBackedEmbeddings
from langchain.globals import set_debug
from langchain.globals import set_verbose
from langchain.llms import LlamaCpp
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware

# Debugging Variables
set_debug(True)
set_verbose(True)


# Load json function
def load_json_file(file_path):
    docs = []
    # Load JSON file
    with open(file_path, encoding="iso-8859-1") as file:
        data = json.load(file)

    # Iterate through 'pages'
    for index, question in data.items():
        q = question['question']
        a = question['answer']
        docs.append(Document(page_content=a, metadata={index: q}))
    return docs


# Define Sources
sources = set()


# Format Docs as string
def format_docs(input_docs):
    content = set()
    for d in input_docs:
        metadata = list(d.metadata)
        for m in metadata:
            sources.add(f'https://www.hawaii.edu/askus/{m}')
        content.add(d.page_content)
    return "\n\n".join(content)


# Path to model
MODEL_FILEPATH = "../../resources/models/llama2Medium.gguf"

# System Prompts
SYSTEM_TEMPLATE = """You are a chatbot. You are assisting a human with a question. Here is some context to help you answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}"""

messages = [SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template("[INST]{question}[/INST]")]

prompt = ChatPromptTemplate.from_messages(messages)

# Define Model
model = LlamaCpp(
    model_path=MODEL_FILEPATH,
    max_tokens=1024,
    n_ctx=4096,
    n_threads=10,
    n_gpu_layers=40,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,  # Verbose is required to pass to the callback manager
)


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )
    question: str


# https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token
# Splitting by character may result in some loss of context? Maybe worth attempting to split by tokens.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,

)

# Embed the text and cache it
underlying_embeddings = GPT4AllEmbeddings()
fs = LocalFileStore("../cache")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, fs, namespace="Token"
)
# Split the documents into smaller chunks
documents = text_splitter.split_documents(load_json_file("../../resources/json/data.json"))

# Store the embedded chunks into a vector store for easy access
vectorstore = Chroma.from_documents(documents, cached_embedder)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever = vectorstore.as_retriever()

# chain = ConversationalRetrievalChain.from_llm(model, retriever, prompt).with_types(
#     input_type=ChatHistory
# )
# Final chain to pipe everything
# chain = (
#         {
#             "context": retriever | format_docs,
#             "question": RunnablePassthrough()
#         }
#         | prompt
#         | model
#         | StrOutputParser()
# )

# FAST API SECTION
app = FastAPI(title="Chatbot", version="1.0")

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Test get with url parameter
@app.get("/help/{question}")
async def root(question):
    a = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
    )
    return f'{a.invoke(question)}\n\nSources: {", ".join(sources)}'


#
#
# @app.post("/test/invoke")
# async def work(query: Question):
#     a = (
#             {
#                 "context": retriever | format_docs,
#                 "question": RunnablePassthrough()
#             }
#             | prompt
#             | model
#             | StrOutputParser()
#     )
#     return f'{a.invoke(query)}\n\nSources: {", ".join(sources)}'


# #
# @app.get("/chain/invoke")
# async def get_chain():
#     return (
#             {
#                 "context": retriever | format_docs,
#                 "question": RunnablePassthrough()
#             }
#             | prompt
#             | model
#             | StrOutputParser()
#     )



# Model without context

# add_routes(app, retriever | format_docs, path="/retriever")
add_routes(app, retriever)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
)
chain = chain.with_types(input_type=Question)

add_routes(app, chain, path="/chain", input_type=Question)
# add_routes(app, chain, path="/llama")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
