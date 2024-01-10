from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub

llm = Ollama(
    base_url='http://localhost:11434',
    model='mistral',
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Define vectostore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

# Define RetrievalQA chain
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")
# print("Prompt:", QA_CHAIN_PROMPT)

chain = RetrievalQA.from_chain_type(
    llm, 
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Define prompt
query = "Who is Neleus and who is in Neleus' family?"
print(f"Query: {query}")
response = chain({"query": query})





