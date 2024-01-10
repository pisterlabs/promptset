from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

# Load environment variables
load_dotenv()

# Instantiate embeddings
open_ai_embeddings = OpenAIEmbeddings()

# Instantiate text splitter
text_splitter = CharacterTextSplitter(separator="\n",
                                      chunk_size=200,
                                      chunk_overlap=0)

# Load facts.txt file
text_loader = TextLoader("facts.txt")
facts_doc = text_loader.load_and_split(text_splitter=text_splitter)

# Instantiate ChromaDB to generate and store the embeddings
chroma_db = Chroma.from_documents(documents=facts_doc,
                                  embedding=open_ai_embeddings,
                                  persist_directory="chroma_db")
