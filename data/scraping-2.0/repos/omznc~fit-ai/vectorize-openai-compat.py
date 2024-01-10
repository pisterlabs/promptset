import os

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

# Definicije putanja i direktorija
documents_directory = "documents"
chroma_persist_directory = "./chroma_openai_compat"

# Učitavanje potrebnih biblioteka i modela
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

print("Starting...")

# Stvaranje prazne liste za pohranu učitanih dokumenata
documents = []

# Koristenje loop-a za iteriranje kroz sve datoteke u direktoriju dokumenata
for filename in os.listdir(documents_directory):
	if not filename.endswith(".pdf"):
		continue
	document = PyPDFLoader(os.path.join(documents_directory, filename)).load_and_split()
	documents.append(document)

if len(documents) == 0:
	print("Dokumenti ne postoje u {}".format(documents_directory))
	exit()

print("Loadano {} dokumenata...".format(len(documents)))

# Djeljenje dokumenata na manje dijelove koristeći CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Kombiniranje svih chunk-ova
all_chunks = []
for document in documents:
	chunks = text_splitter.split_documents(document)
	all_chunks += chunks

# Persistanje Chroma objekta u direktoriju
db2 = Chroma.from_documents(all_chunks, embeddings, persist_directory=chroma_persist_directory)
db2.persist()

print("Done.")
