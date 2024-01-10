from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# find aesthetics treatments on a given web page

# @see https://python.langchain.com/docs/integrations/llms/ollama
# setup:
# ./ollama serve
# ./ollama run llama2
# run: python aesthetics-treatments-rag.py

# SETUP LLM:
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# this uses the local llm web server apis once you have it running via ollma: https://ollama.ai/
llm = Ollama(
   model="llama2:13b", 
   callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)




# VECTORDB-IZE WEB DATA
# pages = ["https://www.medicalaestheticclinic.co.uk/treatments"]
pages = ["https://www.epsomskinclinics.com/"] # epsom skin clinic
# pages = ["https://www.altondental.co.uk/"]
print("data sourced from following web pages: ", pages)
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
prompt = PromptTemplate.from_template(template)

# LLM Query Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
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
question = "bullet list all aesthetics treatments found in documents"
result = chain.invoke(question)





