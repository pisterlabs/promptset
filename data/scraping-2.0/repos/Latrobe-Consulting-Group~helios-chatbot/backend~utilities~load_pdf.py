# load_pdf.py - Loads PDF documents into the LanceDB vector store

## Imports:
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB
import dotenv
import lancedb
import os
from pypdf import PdfReader

## Set Env Variables
dotenv.load_dotenv()
if None == os.environ.get('OPENAI_API_KEY'):
    raise ValueError("Env OPENAI_API_KEY not set")
else:
    OAI_TOKEN = os.environ.get('OPENAI_API_KEY')

filename = "./backend/utilities/NAME_OF_PDF.pdf"

## Set up connection to OpenAI
embeddings = OpenAIEmbeddings()

## Set up knowledge VectorStore
db_name = "./helios_kb.db"
table_name = "helios_kb"
db = lancedb.connect(db_name)
if table_name not in db.table_names():
    table = db.create_table(
        "helios_kb",
        data=[
            {
                "vector": embeddings.embed_query("You are Helios, an AI chatbot that can perform background research tasks."),
                "text": "You are Helios, an AI chatbot that can perform background research tasks with access to the internet.",
                "id": "1",
            }
        ],
        mode="create",
    )
else:
    table = db.open_table(table_name)
vectorstore = LanceDB(connection=table, embedding=embeddings)

## Load and split PDF file
reader = PdfReader(filename)
parts = []

def visitor_body(text, cm, tm, font_dict, font_size):
    y = cm[5]
    if y > 50 and y < 720:
        parts.append(text)

page_id = 0
for page in reader.pages:
    print("Loading Page " + str(page_id) + "...")
    text = page.extract_text(visitor_text=visitor_body)
    text_splitter = CharacterTextSplitter(        
        separator = "\n\n",
        chunk_size = 2500,
        chunk_overlap  = 250,
        length_function = len,
        is_separator_regex = False,
    )
    docs = text_splitter.create_documents([text])
    for doc in docs:
        vectorstore.add_texts(texts=[text], metadatas=[{"filename": filename, "page_number": page_id}])
    page_id += 1