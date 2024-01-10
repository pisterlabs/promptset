# process all documents into embeddings
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# converts PDF to text and splits data per page
from langchain.document_loaders import UnstructuredPDFLoader
def extract_pdf_text(pdf_path):
    loader = UnstructuredPDFLoader(pdf_path, mode="elements") # this is taking very long for some reason -> need to find alternative, it does seem to split text better than PyPDFLoader tho
    data = loader.load() # load and load_and_split are the same for unstructuredPDFLoader
    pdfname = pdf_path.split('/')[-1]
    # for each element of data list, set the source key as the pdf name
    for x in range(len(data)):
        data[x].metadata['source'] = pdfname
    return data

# takes big chunk of text and splits it down to smaller documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
def divide_pdf_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #can also add seperators paramater like  separators=["\n\n", "\n", " ", ""]
    texts = text_splitter.split_documents(data)
    return texts

# converts each document into a vector using OpenAI embeddings and stores these vectors externally in Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
def create_embeddings(texts):
    embeddings = OpenAIEmbeddings() #might want to specify model name but it looks like only ada is available for embeddings
    pinecone.init(
            api_key=os.environ['PINECONE_API_KEY'],
            environment=os.environ['PINECONE_ENVIRONMENT']  
        )
    index_name = "user1" # index needs lowercase, or "-", and must start and end with alphanumeric character 
    namespace = "combined_pdfs" 

    upsertChunkSize = 50 # pinecone recommends a limit of 100 vectors per upsert request to avoid errors
    print("number of vectors: ", len(texts))
    for i in range(0,len(texts), upsertChunkSize):
        chunk = texts[slice(i, i+upsertChunkSize)]
        print("length of chunk: ",len(chunk))
        Pinecone.from_documents( #using from_documents instead of from_texts - idk why
            chunk, 
            embeddings,
            index_name=index_name) #removed namespace for now because JS version of langchain does not let you query based on namespace for some reason
        
# iterate through each document and extract text, split text, and create embeddings
def process_data():
    combinedData = []
    for root, dirs, files in os.walk("/Users/sheelsansare/Desktop/Spect-AI/public/LakersPracticeFacility"):
        for filename in files:
        # WARNING
        # this wont work anymore because the folder has been moved
            pdfpath = os.path.join(root,filename)
            # manually skip over .DS_Store file
            if pdfpath == "/Users/sheelsansare/Desktop/Spect-AI/public/LakersPracticeFacility/.DS_Store":
                continue
            if pdfpath != "/Users/sheelsansare/Desktop/Spect-AI/public/LakersPracticeFacility/Specs/2016_0115 Specification.pdf":
                continue
            print(pdfpath)
            data = extract_pdf_text(pdfpath)
            combinedData = combinedData + data
    texts = divide_pdf_text(combinedData)   
    create_embeddings(texts)
        
if __name__ == "__main__":
    start = time.time()
    process_data()
    end = time.time()
    print(end-start, " seconds")