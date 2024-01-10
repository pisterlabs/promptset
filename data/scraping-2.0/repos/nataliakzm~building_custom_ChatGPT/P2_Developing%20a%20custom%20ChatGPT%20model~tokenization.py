# Import necessary modules
import tiktoken
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Specify the model to use and get the appropriate encoding
tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

# Load an unstructured PDF file
loader = UnstructuredPDFLoader('/content/yourdocument.pdf')
data = loader.load()

# Define a function to get token length
def tiktoken_len(text):
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)

# Split document into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20,
         length_function=tiktoken_len, separators=["\n\n", "\n", " ", ""])

# Split the loaded document into chunks
texts = text_splitter.split_documents(data)
