# Embed the mdx files to a vectorstore as embedding using cohere embedding model
# 1. Get the mdx file
# 2. Convert this mdx file and vectorize those
# 3. Get vector Embeddings
# 4. Add the embedding to a vector DB
# 5. Build a user chat interface so that we can get the query.
# 6. Use reranker to get most relevant results from the vectorDB
# 7. For each query from the user, give the query to rerank and take the result from the rerank and supply it to Gemini.


import os
import re
import cohere
import json
import cassio
from dotenv import load_dotenv
from embedAstra import embedIntoAstra
# Llama_index compoenent to use
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file.markdown_reader import MarkdownReader

# LangChain components to use
from langchain.agents import Tool
from langchain.document_loaders import TextLoader




load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)
base_path = './'

def save_documents_to_file(documents, filename):
    """# Save the extracted documents to a new folder/file"""
    # Create a folder
    folder_path = os.path.join(base_path, filename)
    os.makedirs(folder_path, exist_ok=True)
    # save the chunks to that folder.
    with open(folder_path, 'w') as file:
        for document in documents:
            file.write(document + '\n')

    print("Saving Exectued !")


def get_fileName(documents):
    """# To get the name of the document extracted """
    fileName = ""
    for doc in documents:
        fileName = doc.metadata['file_name']
    print(fileName)
    return fileName


def get_cleaned_file_names(documents):
    """Used to create the file extension to save the file"""

    for doc in documents:
        original_file_name = doc.metadata.get('file_name', '')
        # Split the file name by '.' to remove the extension
        file_name_parts = original_file_name.split('.')
        # Take only the first part (without the extension)
        base_file_name = file_name_parts[0]
        # Append ".txt" to the cleaned file name
        cleaned_file_name = base_file_name + '.txt'
    return cleaned_file_name


def formatToEmbed(document):
    """To convert the extracted documents into format that can be embedded."""

    toEmbed = []
    for doc in document:
        # print(doc.text)
        docToString = '"""' + doc.text + '"""'
        # toEmbed.append(doc.text)
        toEmbed.append(docToString)
    return docToString


def embedDocuments(documents):
    """Using cohere embedding to embed documents and using formatToEmbed to convert the data into list of string"""

    response = co.embed(
        texts=documents,
        input_type="search_document",
        model="embed-english-v3.0",
    )

    # print(response)
    print('Embeddings: {}'.format(response.embeddings))
    return response.embeddings


def loadFile(required_ext):
    """To extract the documents from the file"""

    # required_exts = [required_ext]

    # reader = SimpleDirectoryReader(
    #     input_files=["./accentColor.mdx"],
    #     # input_dir="./tailwindcss/src/pages/docs",
    #     file_extractor={"mdx": MarkdownReader},
    #     # required_exts=required_exts,
    #     recursive=True
    # )
    # return reader.load_data()
    docs = []
    directory_path = "./"
# Loop through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(required_ext):
            file_path = os.path.join(directory_path, filename)
            
            # Assuming your TextLoader class takes a file path as an argument
            loader = TextLoader(file_path)
            
            # Load the document and append it to the docs list
            document = loader.load()
            docs.append(document[0])

    return docs



def prettyPrint(document):
    """Pretty print or simply print the extraccted documents"""
    for doc in document:
        print(doc.text)
        print(len(doc.text))

    fileName = get_cleaned_file_names(document)
    print(fileName)


def vectorIndex():
    """ A vector index enables us to find the specific data we are looking 
    for in large sets of vector representations easily. ( vector indexing ).
    """
    # Default session index
    # index = VectorStoreIndex.from_documents(documents=vectorEmbeddings)


def getTools(toolName,index):
    """Tools to give to agent.
        Can be used to retrieve documents as per our need. Using a indexer to query and search.
        retrieve : Tool to retrieve relevant documents from store.
    """
    if toolName == "retrieve":
        tools = [
            Tool(
                name="TAILWIND",
                func=lambda q: str(index.as_query_engine().query(q)),
                description="useful for when you want to answer questions about TAILWIND. The input to this tool should be a complete english sentence.",
                return_direct=True,
            ),
        ]
    return tools


def main():

    # document = formatToEmbed(loadFile(".md"))
    # prettyPrint(document=document)
    # print(document)
    
    docs = loadFile(".mdx")
    # print(docs)
    # vectorEmbeddings = embedDocuments(document)
    # json_str = json.dumps(obj.to_json())
    # embedIntoAstra(json.dumps(document.to_json()))
    embedIntoAstra(docs,"")
    # print(document)
    # embedIntoAstra(document,vectorEmbeddings)


if __name__ == "__main__":
    main()
