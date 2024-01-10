import os 
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

import pinecone

def readFile(path):
    """Converts the PDF from Path to a single String - delimited by \n"""
    # Read the file
    readData = PyPDF2.PdfReader(path)

    # Initiate the text 
    text = ""

    for page in readData.pages:
        if page:
            text += (page.extract_text() + "\n")

    print(f"Length of text : {len(text)}")
    return text

def splitData(data):
    """Splits the data into chunks of 500 words each"""

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 50,
    length_function = len,
    add_start_index = True,
)
    texts = text_splitter.create_documents([data])
    texts = [ text.page_content for text in texts]
    print(f"Length of texts : {len(texts)}")
    return texts


def convertToEmbeddings(data):
    # Create the embedding model from OpenAI
    key = os.environ.get('OPENAI_KEY')
    print(f"Key : {key}")
    embeddingModel = OpenAIEmbeddings(openai_api_key = key)

    # Convert the data to embeddings - we expect a list of strings as input (Data)
    embeddings = embeddingModel.embed_documents(data[0])

    print(f"Length of embeddings : {len(embeddings)}")
    print(f"Length of each Embedding : {len(embeddings[0])}")
    return embeddings

def createVectorIndex():

    pinecone.create_index("contextdatascience", dimension=1536, metric="euclidean")
    pinecone.describe_index("contextdatascience")

def converListToDict(data, embeddings):
    dictData = []
    numData = len(data)

    for i in range(numData):
        dictData.append({"id": str(i), "values": embeddings[i], "metadata":{"values": data[i]}})

    return dictData

def storeToVectorDB(data, embeddings):
    pinecone.init(api_key=os.environ.get('PINECONE_KEY'), environment='gcp-starter')

    # Create the index
    #createVectorIndex()

    index = pinecone.Index("contextdatascience")

    # Insert the embeddings
    datadict = converListToDict(data, embeddings)
    index.upsert(datadict)

    return "Success"

def main():
    pathToFile = "Docs/50YearsDataScience.pdf"

    textString = readFile(pathToFile)
    chunks = splitData(textString)
    embeddings = convertToEmbeddings(chunks)
    status = storeToVectorDB(chunks, embeddings)

    #print(status)

if __name__ == "__main__":
    main()


class SendDocToPineCone:
    def __init__(self, pineConeKey, openAIKey):
        self.Pkey = pineConeKey
        self.Okey = openAIKey

    def storeToVectorDB(self, data, embeddings, namespace):
        pinecone.init(api_key=self.Pkey, environment='gcp-starter')

        # Create the index
        #createVectorIndex()

        index = pinecone.Index("contextdatascience")

        # Insert the embeddings
        datadict = converListToDict(data, embeddings)
        index.upsert(datadict, namespace=namespace)

        return "Success"

    def convertUploadedFileToEmbeddings(self, file):

        embeddingModel = OpenAIEmbeddings(openai_api_key = self.Okey)
        embeddings = []

        try :
            embeddings = embeddingModel.embed_documents(file[0])
            print(f"Length of embeddings : {len(embeddings)}")
            print(f"Length of each Embedding : {len(embeddings[0])}")

        except:
            print("There was an error in converting the file to embeddings")

        return embeddings

    def sanitizeFileName(self, filename):
        # Convert the filename to lowercase
        filename_lower = filename.lower()
        
        # Use regular expression to remove non-alphanumeric characters
        # This regex will keep lowercase letters, numbers, and the dot character (for file extension)
        sanitized_filename = re.sub(r'[^a-z0-9\.]', '', filename_lower)
        return sanitized_filename
    
    def uploadtoPineCone(self, documents):
        
        for document in documents:
            sanitized_filename = self.sanitizeFileName(document.name)
            file = readFile(document)
            chunks = splitData(file)
            embeddings = self.convertUploadedFileToEmbeddings(chunks)
            status = self.storeToVectorDB(chunks, embeddings, sanitized_filename)

        pass



