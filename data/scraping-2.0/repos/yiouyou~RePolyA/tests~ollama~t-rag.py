_url = "http://localhost:11434"
# _model = "llama2"
_model = "yi:34b-q4_K_M"
# _model = "llama2-chinese:13b-chat-q4_K_M"

from langchain.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import (
    OllamaEmbeddings,
    GPT4AllEmbeddings,
    OpenAIEmbeddings,
)
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.schema import LLMResult


# llm = Ollama(
#     model=_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )
# result = llm("今天星期几？")
# print(result)


# ### embed
# oembed = OllamaEmbeddings(base_url=_url, model=_model)
# result = oembed.embed_query("Llamas are social animals and live with others as a herd.")


### RAG
loader = WebBaseLoader("https://www.gmzyjc.com/read/shl/shl02.05-0.2.0.0.0.html")
data = loader.load()
print(f"Loaded {len(data)} documents")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)
print(f"Split into {len(all_splits)} chunks")

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings()
)

### Retrieve
# question = "How can Task Decomposition be done?"
# docs = vectorstore.similarity_search(question)
# print(f"Retrieved {len(docs)} documents")

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
# QA_CHAIN_PROMPT = """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
# Question: {question} 
# Context: {context} 
# Answer: [/INST]
# """
llm = Ollama(
    model=_model,
    temperature=0,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
print(f"Loaded LLM model {llm.model}")
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
question = "为什么说'发于阴，六日愈'？"
result = qa_chain({"query": question})
print(result)

