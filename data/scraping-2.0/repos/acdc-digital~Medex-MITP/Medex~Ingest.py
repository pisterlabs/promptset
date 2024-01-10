import os
import json
from tqdm import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredAPIFileLoader
from langchain.vectorstores import MyScale, MyScaleSettings

# Set API keys as environment variables for security
os.environ['OPENAI_API_KEY'] = "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"
os.environ['MYSCALE_API_KEY'] = "6B71NumcMB7QXcguTapGBjCEWqM27p"

# Configure MyScale settings
config = MyScaleSettings(host='msc-3f5d0ca4.us-east-1.aws.myscale.com', port=443, username='smatty662', password='passwd_CAdIn9GSXH7GNt')
index = MyScale(OpenAIEmbeddings(), config)

# Initialize LlamaIndex components
embed_model = OpenAIEmbeddings()

def process_files(directory):
    # Initialize an empty list to hold file loaders
    file_loaders = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Create a file loader for the PDF file
            # The file loader uses the MyScale API key and a 'fast' strategy
            file_loader = UnstructuredAPIFileLoader(
                file_path,
                api_key=os.environ['MYSCALE_API_KEY'],
                strategy="fast",  # use the 'fast' strategy
                request_kwargs={"timeout": 600}  # set a timeout of 600 seconds
            )

            # Add the file loader to the list
            file_loaders.append(file_loader)

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5500, chunk_overlap=1000)

    # Wrap your file_loaders with tqdm for progress bar
    # Loop through each file loader and process the file
    for loader in tqdm(file_loaders, desc="Processing files"):
        # Load the document
        doc = loader.load()

        # Split the document into chunks
        docs = text_splitter.split_documents(doc)

        # Initialize an empty dictionary to hold the embeddings
        doc_embeddings = {}

        # Loop through each chunk
        for i, d in enumerate(docs):
            # Set the source metadata to the file path
            d.metadata = {"source": loader.file_path}

            # Generate embeddings for the chunk and store them in the dictionary
            doc_embeddings[i] = embed_model.embed_documents([d.page_content])

            # Add the chunk to the LlamaIndex
            index.add_documents([d])

        # Write the embeddings to a JSON file
        with open(f"{loader.file_path}.json", "w") as f:
            f.write(json.dumps(doc_embeddings, default=str))

def retrieve_from_index(query):
    # Retrieve documents from the index based on the query
    results = index.retrieve_documents(query)

    # Return the results
    return results

# Directory containing PDF files to process
directory = "/Users/matthewsimon/Documents/GitHub/Medex-Public-MITP/Medex-Public-MITP/Medex/Source_Documents"
# Call the process_files function to process all PDF files in the directory
process_files(directory)
