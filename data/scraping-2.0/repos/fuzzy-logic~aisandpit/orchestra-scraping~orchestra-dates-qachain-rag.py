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



# Example of finding concert date/time/location in a given web page
# using a LLM specific Q/A chain @see  https://smith.langchain.com/hub/rlm/rag-prompt-llama 
# Typically more of a chatbot conversation 
# @docs https://python.langchain.com/docs/integrations/llms/ollama


# ISSUES
# most pages have side bars or footer with ltos of other events and event dates which seem to confuse the LLM
# we will need to find a way to spearate out the core page/hero content and remove peripheral content or ads 

# setup:
# ./ollama serve
# ./ollama run llama2
# run: python orchestra-dates-qachain-rag.py

# this uses the local llm web server apis once you have it running via ollma: https://ollama.ai/
llm = Ollama(
    model="llama2:13b",
    verbose=False,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# VECTORDB-IZE WEB DATA
pages = ["https://www.rpo.co.uk/whats-on/eventdetail/1982/82/john-rutters-christmas-celebration-matinee"];
print("data sourced from following web pages: ", pages)
all_splits = [];
for page in pages: 
    loader = WebBaseLoader(page)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_splits  = [*all_splits, *text_splitter.split_documents(data)];
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())


# rag qa prompt info: https://smith.langchain.com/hub/rlm/rag-prompt-llama
# changing this prompt will radically change the behavior of the llm
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# Run: this prompt is the instruction:
# multi event list Prompt: "List all performance events, include name, time, location, next performance date and any supplimental information that is provided"
# simple primary event prompt: "List the primaray performance event information. Include name, time, location, next performance date and any supplimental information that is provided"
question = "Provide a bullet list of the primaray performance event name, date, time, location and supplimental information"
qa_chain({"query": question})






