import os
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import dotenv
from pathlib import Path, PurePath
import PyPDF2

dotenv.load_dotenv()

# enter the document name for which vector to be created
document_name = str(input('Enter the document name for which vector to be created(keep it short): '))
# pdf to text

pdfFileObj = open('PDP document for QA bot_v1.pdf', 'rb')
pdfReader = PyPDF2.PdfReader(pdfFileObj)
num_pages = len(pdfReader.pages)
print(pdfReader.pages)
data = []
for page in range(0, num_pages):
    print('print', page)
    pageObj = pdfReader.pages[page]
    page_text = pageObj.extract_text()
    data.append(page_text)
pdfFileObj.close()
data = data[0:15]  # Number of page for which vector is created

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    #print(i, len(splits))
    docs.extend(splits)
    #metadatas.extend([{"source": sources[i]}] * len(splits))

metadatas = [{"source":"PDP DOCUMENTATION INDEX"}, {"source":"SUPPORT"},{"source":"API INDEX BY TYPE"},
    {"source":"INTRO TO PDP"},{"source":"How PDP differs from After-market devices?"},
    {"source":"PDP’s APIs RePEAT"}, {"source":"Quick brief about GraphQL"},{"source":"GraphQL Methods"},
    {"source":"Modules"}]

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
# Get the directories for vector store
root_path = PurePath(Path(__file__).parents[1]).as_posix()
vector_path = os.path.join(root_path, 'application', 'vectorstores', 'tvs', f'{document_name}')
os.makedirs(vector_path, exist_ok=True)
# write docs.index and pkl file
faiss.write_index(store.index, os.path.join(vector_path,"docs.index"))
store.index = None
with open(os.path.join(vector_path,"faiss_store.pkl"), "wb") as f:
    pickle.dump(store, f)