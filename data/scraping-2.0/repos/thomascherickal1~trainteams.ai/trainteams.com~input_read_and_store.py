import regex as re
from PyPDF2 import PdfReader
from langchain.vectorstores import DeepLake
import spacy
from langchain.embeddings import SentenceTransformerEmbeddings
pdfFile = open('Rust Programming.pdf', 'rb')
reader = PdfReader(pdfFile)

document = ""

print(len(reader.pages))

for pageNum in range(len(reader.pages)):
    page = reader.pages[pageNum].extract_text()
    document += page

document = document.lower()

# Step 2: Remove punctuation
document = re.sub(r'[^\w\s]', '', document)

nlp = spacy.load("en_core_web_sm")
doc = nlp(document)
tokens = [token.text for token in doc]
embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')


# Define your documents and embedding function
docs = tokens # the documents need to be a list of lists of words
# Create and persist the vector store
db = DeepLake.from_texts(docs, embedding_function, dataset_path="./deeplake_db")




