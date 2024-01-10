# E. Culurciello, June 2023
# test langchain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# from searchapi import getSearchData

# search data on web:
query = "What is the name of the black hole in the Milky Way galaxy?"
# data = getSearchData(query, num_results=1)
# print(data)

loader = TextLoader("data.txt")
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)
print(texts)
docsearch = Chroma.from_texts(texts, embeddings)

# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# # Verbose is required to pass to the callback manager
# llm = LlamaCpp(
#     model_path="./vicuna-7b-1.1.ggmlv3.q4_0.bin", 
#     callback_manager=callback_manager, 
#     verbose=True
# )

# qa = RetrievalQA.from_chain_type(
#         llm=llm, 
#         chain_type="map_reduce", 
#         retriever=docsearch.as_retriever()
#     )

# qa.run(query)
