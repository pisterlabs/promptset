from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI
import json
import pinecone
import keyring
import os
import warnings
import math
import re

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.beam_search")

# Get the API key from the system's keyring
api_key = keyring.get_password("openai", "api_key")

# Check if the API key was retrieved successfully
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    print("Failed to retrieve the API key.")

# Define the chunk and overlap sizes
CHUNK_SIZE = 1000  # Adjust this value to your needs
OVERLAP = 100  # Adjust this value to your needs
DIMENSION_VALUE = 1536 # Number of dimentions used for embedding

# Initialize Pinecone
pinecone.init(api_key="9e656193-e394-43af-8147-5dcc62a22ef2", environment="asia-southeast1-gcp-free")

input_directory = "./test"
output_directory = "./output"

def find_largest_page_number(text):
    # Search for all occurrences of the page boundary pattern
    matches = re.findall(r'--- End of Page (\d+) ---', text)
    
    # If no matches found, return -1
    if not matches:
        return -1
    
    # Convert the matches to integers and return the maximum value
    return max(map(int, matches))

def overlap_length(s1, s2):
    overlap = 0
    # Check if s1 is a suffix of s2 or vice-versa
    for i in range(min(len(s1), len(s2))):
        if s1[:i+1] == s2[-i-1:] or s1[-i-1:] == s2[:i+1]:
            overlap = i + 1
    return overlap

def parse_docs_to_nodes(documents, directory):
    parser = SimpleNodeParser.from_defaults(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    nodes = []

    for doc in documents:  # Here, doc is an instance of ImageDocument
        doc_name = doc
        content = None
        with open(input_directory + "/" + doc, 'r') as file:
            content = file.read()
        doc_content = [Document(text=content)]

        # Read the JSON file to get the page_number
        with open(directory + '/' + doc_name.replace('.txt', '.json'), 'r') as json_file:
            metadata = json.load(json_file)
            total_pages_from_json = metadata.get('page_number', 0)

        doc_nodes = parser.get_nodes_from_documents(doc_content)
        previous_largest = 1
        for i, node in enumerate(doc_nodes):
            previous_largest = max(find_largest_page_number(node.text), previous_largest)
            node.metadata = {
                "document_name": doc_name,
                "page_number": previous_largest
            }
            nodes.append(node)
            #previous_node_content = node.text

    # Assuming nodes is your collection of nodes
    previous_node = nodes[0]
    for node in nodes[1:]:
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=previous_node.node_id, metadata={"key": "val"})
    return nodes


def serialize_text_node(node):
    return {
        "text": node.text,
        "id_": node.id_,
        "document_name": node.metadata["document_name"],
        "page_number": node.metadata["page_number"]
    }

def store_nodes_to_json(nodes, directory):
    # Convert to JSON string\
    serialized_nodes = [serialize_text_node(node) for node in nodes]
    json_string = json.dumps(serialized_nodes, indent=4)  # indent=4 for pretty-printing

# Write to file
    with open(directory + "/nodes.json", "w") as file:
     file.write(json_string)

# Create or connect to a Pinecone index
# This function creates a Pinecone index with the given name and dimension, using a single shard.
def create_index(index_name, index_dimension):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=index_dimension)
    else:
        print(f"Index {index_name} already exists.")

def get_index(index_name, index_dimention):
    create_index(index_name, index_dimention)
    return pinecone.Index(index_name=index_name)

def chunk_and_vectorize():
    print("Please select an operation:")
    print("1. Create Node.json only")
    print("2. Create index in Pinecone vector DB only")
    print("3. Do both")
    
    choice = input("Enter your choice (1/2/3): ")

    files = os.listdir(input_directory)
    # To only get files and exclude directories
    only_files = [f for f in files if os.path.isfile(os.path.join(input_directory, f))]
    txt_files = [f for f in only_files if f.endswith('.txt')]
    print(txt_files)

    #all_documents = SimpleDirectoryReader(input_directory).load_data()
    #print(all_documents)
    #documents = [doc for doc in all_documents if doc.endswith('.txt')]
    nodes = parse_docs_to_nodes(txt_files, input_directory)

    if choice == "1" or choice == "3":
        store_nodes_to_json(nodes, output_directory)
        print("Node.json created.")

    if choice == "2" or choice == "3":
        pinecone_index = get_index("history-chunks", DIMENSION_VALUE)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, llm=OpenAI())
        index = VectorStoreIndex(
            nodes, 
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True
        )
        index.set_index_id("vector_index")
        index.storage_context.persist("./index")
        print("Index created in Pinecone vector DB.")

# Call the function with your text
chunk_and_vectorize()


