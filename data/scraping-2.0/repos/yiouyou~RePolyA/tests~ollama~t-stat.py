_url = "http://localhost:11434"
# _model = "llama2"
# _model = "yi:34b-q4_K_M"
_model = "llama2-chinese:13b-chat-q4_K_M"

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


### RAG
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings())
question = "How can Task Decomposition be done?"
docs = vectorstore.similarity_search(question)
print(len(docs))

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
# QA_CHAIN_PROMPT = """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
# Question: {question} 
# Context: {context} 
# Answer: [/INST]
# """
### log tokens
class GenerationStatisticsCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(response.generations[0][0].generation_info)
callback_manager = CallbackManager(
    [StreamingStdOutCallbackHandler(), GenerationStatisticsCallback()]
)
llm = Ollama(
    base_url=_url,
    temperature=0,
    model=_model,
    verbose=True,
    callback_manager=callback_manager,
)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
question = "What are the approaches to Task Decomposition?"
result = qa_chain({"query": question})
print(result)

