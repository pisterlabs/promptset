from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# load the document and split it into chunks
from pdfminer.high_level import extract_text


pdf_files_directory = "docs"  # Directory containing the PDF files

# Create a list to store the extracted data from each PDF
data = []

# Loop through all the files in the directory
for index, filename in enumerate(os.listdir(pdf_files_directory)):
    # Check if the file is a PDF
    if filename.lower().endswith(".pdf"):
        full_path = os.path.join(pdf_files_directory, filename)
        print(f"Loading {full_path}")

        # Load the PDF and extract its text
        text = extract_text(full_path)

        # Create a dictionary to store the document information
        document_info = {
            "document_id": index,  # Unique identifier for the document
            "document_name": filename,  # Name of the document
            "page_content": text,
            "metadata": {
                "source": full_path,
                "page": len(data)  # Page number is the index of the document in 'all_data' list
            }
        }

        # Append the document information to the list
        data.append(document_info)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

class DocumentData:
    def _init_(self, page_content, metadata, document_id, document_name):
        self.page_content = page_content
        self.metadata = metadata
        self.document_id = document_id
        self.document_name = document_name
# Assuming 'all_data' contains the dictionaries as created earlier
all_documents = []
for document_info in data:
    page_content = document_info["page_content"]
    metadata = document_info["metadata"]
    document_id = document_info["document_id"]
    document_name = document_info["document_name"]

    document = DocumentData(page_content=page_content, metadata=metadata, document_id=document_id, document_name=document_name)
    all_documents.append(document)
   
    docs = text_splitter.split_documents(all_documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# Split documents into chunks using the initialized text splitter
docs = text_splitter.split_documents(docs)

# Create the open-source embedding function
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load documents into Chroma
db = Chroma.from_documents(docs, embeddings)
def docsearch_with_name_similarity(query, documents):
    query_with_names = query + " " + " ".join([doc.document_name for doc in documents])
    print(query_with_names)
   
    return db.similarity_search(query_with_names)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

"""#  Quantized Models from the Hugging Face Community"""

model_path = "llama-2-13b-chat.ggmlv3.q5_1.bin"
n_gpu_layers = 40
n_batch = 256

# Load LlamaCpp model
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    n_ctx=1024,
    verbose=False,
)

chain = load_qa_chain(llm, chain_type="stuff")

# Query loop
while True:
    query = input("Enter your question (or 'q' to quit): ")
    if query.lower() == 'q':
        break
    docs = docsearch_with_name_similarity(query, all_documents)
    chain.run(input_documents=docs, question=query)
    print("\n")