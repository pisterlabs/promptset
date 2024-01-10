from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.chains import RetrievalQA
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

# Example of using LLM + RAG with vector database and simple prompt chain
# @see https://research.ibm.com/blog/retrieval-augmented-generation-RAG

# @docs https://python.langchain.com/docs/integrations/llms/ollama
# setup:
# ./ollama serve
# ./ollama run llama2
# run: python orchestra-dates-rag.py

# ISSUES
# most pages have side bars or footer with ltos of other events and event dates which seem to confuse the LLM
# we will need to find a way to spearate out the core page/hero content and remove peripheral content or ads 




### VECTORDB-IZE THE WEB DATA
pages = ["https://www.rpo.co.uk/whats-on/eventdetail/1982/82/john-rutters-christmas-celebration-matinee"];
print("following data sourced from following web pages: ", pages)
for page in pages: 
    loader = WebBaseLoader(page)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits  = text_splitter.split_documents(data);
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())



### SETUP THE PROMPT CHAIN:
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following documents:
{docs}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# this uses the local llm web server apis once you have it running via ollma: https://ollama.ai/
llm = Ollama(
    model="llama2:13b",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
chain = (
    {"docs": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

### FIRE OFF QUESTION
question = "Provide a bullet list of performance event name, time, date, prices, location"
result = chain.invoke(question)
print(result)






