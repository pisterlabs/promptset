
# (Ensure these imports are aligned with the latest Chroma and LangChain documentation)
from dotenv import load_dotenv 
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import os

# Load environment variables
load_dotenv()

# Prepare directories for DB
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")

# Load the document
doc_loader = TextLoader('./aikit/content/tool_concept.txt', encoding="utf8")
document = doc_loader.load()

# Split the document data
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
split_docs = text_splitter.split_documents(document)

# Load the embeddings from OpenAI
openai_embeddings = OpenAIEmbeddings()

# Configure the new Chroma database settings
# (Ensure these settings are consistent with the new Chroma version)
client_settings = Settings(
    # Update these settings based on the latest Chroma configuration requirements
)


# Check if the database exists and create or load accordingly
if not os.path.exists(DB_DIR):
    vector_store = Chroma.from_documents(
        split_docs,
        openai_embeddings,
        persist_directory=DB_DIR,
        client_settings=client_settings,
        collection_name="transcripts_store"
    )
    vector_store.persist()
else:
    vector_store.load_collection("transcripts_store")

# Create and configure the QA chain
qa_with_source = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Define a function to query the document
def query_document(question: str) -> dict[str, str]:
    return qa_with_source({"question": question})

# Interactive querying loop
while True:
    print("What is your query? ", end="")
    user_query = input("\033[33m")
    print("\033[0m")
    if user_query == "quit":
        break
    response = query_document(user_query)
    print(f'Answer: \033[32m{response["answer"]}\033[0m')
    print(f'Sources: \033[34m{response["sources"]}\033[0m')
