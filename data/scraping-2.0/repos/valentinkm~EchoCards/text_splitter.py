from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader


# Recursive splitting to consider different separators in generic text
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200, 
    separators=["\n\n", "\n", " ", "", "."],
    length_function = len
)
transcript_file = "files/transcript.txt"

# create function of the above
def load_split_transcript(transcript_file):
    loader = TextLoader(transcript_file)
    data = loader.load()
    docs = r_splitter.split_documents(data)
    return docs
