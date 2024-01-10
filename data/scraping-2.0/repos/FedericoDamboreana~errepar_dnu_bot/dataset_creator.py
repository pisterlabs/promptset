from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import docx

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print(">>> model loaded")

doc = docx.Document("store/dataset.docx")
print(">>> document loaded")

document_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ''])
print(">>> text extracted: ", document_text)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=400,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
documents = text_splitter.create_documents([document_text])
docs = text_splitter.split_documents(documents)
chroma_db = Chroma.from_documents(docs, embedding_function, persist_directory="./store")