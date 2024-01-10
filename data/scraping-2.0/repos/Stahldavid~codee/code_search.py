import code_embedding


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


persist_directory = 'db_ros2_control'

embeddings = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb_wr2 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

query = "Variable impedance controlfor force feedback"
codes = vectordb_wr2.similarity_search(query)

print(codes)