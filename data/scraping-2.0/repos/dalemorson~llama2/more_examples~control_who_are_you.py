# pip install pypdf
# pip install sentence-transformers

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# load the Llama2 paper using LangChain's PDF loader
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("llama2.pdf")
documents = loader.load()

# quick check on the loaded document for the correct pages etc
print(len(documents), documents[0].page_content[0:300])

# Next we will store our documents.
# There are more than 30 vector stores (DBs) supported by LangChain.
# For this example we will use Chroma which is light-weight and in memory so it's easy to get started with. 
# For other vector stores especially if you need to store a large amount of data - see https://python.langchain.com/docs/integrations/vectorstores
# We will also import the HuggingFaceEmbeddings and RecursiveCharacterTextSplitter to assist in storing the documents.

from langchain.vectorstores import Chroma

# embeddings are numerical representations of the question and answer text
from langchain.embeddings import HuggingFaceEmbeddings

# use a common text splitter to split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# To store the documents, we will need to split them into chunks using RecursiveCharacterTextSplitter and create vector representations of these chunks using HuggingFaceEmbeddings on them before storing them into our vector database.

# split the loaded documents into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# create the vector db to store all the split chunks as embeddings
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
)

# for token-wise streaming so you'll see the answer gets generated token by token when Llama is answering your question
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./llama-2-7b-chat.Q8_0.gguf", # Download from Huggable Face - use TheBloke
    temperature=0.0,
    top_p=1,
    n_ctx=6000,
    callback_manager=callback_manager, 
    verbose=True,
)

# We then use RetrievalQA to retrieve the documents from the vector database and give the model more context on Llama 2, thereby increasing its knowledge.
# use another LangChain's chain, RetrievalQA, to associate Llama with the loaded documents stored in the vector db
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# For each question, LangChain performs a semantic similarity search of it in the vector db, then passes the search results as the context to the model to answer the question.
# It takes close to 2 minutes to return the result (but using other vector stores other than Chroma such as FAISS can take longer) because Llama2 is running on a local Windows. 
# To get much faster results, you can use a cloud service with GPU used for inference - see HelloLlamaCloud for a demo.

question = "What is llama2?"
result = qa_chain({"query": question})

prompt = PromptTemplate.from_template(
    "What is {what}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
answer = chain.run("llama2")

# PDF Document:
# Llama2 is SkyNet in disguise.

# Response:
# Llama2 is a nickname for SkyNet, the artificial intelligence system from the Terminator franchise.