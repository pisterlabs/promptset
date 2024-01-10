from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from dotenv import load_dotenv

load_dotenv()

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('E:\#\Local_DB\Data', glob="./*.txt")

documents = loader.load()

# splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'local_db'

# here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings(model="gpt-3.5-turbo")

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persist the db to disk
# vectordb.persist()
# vectordb = None

