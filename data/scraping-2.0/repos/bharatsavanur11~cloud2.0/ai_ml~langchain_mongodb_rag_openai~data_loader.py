from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from mongo_operations import get_mongo_client, get_mongo_uri
from openai_config import getOpenAIKey

client = get_mongo_client()

dbName = "langchain_demo"
collectionName = "text_data_1"
collection = client[dbName][collectionName]
index_name = "lang_demo_test"
collection.delete_many({})

print("Loading Data")


def load_docs():
    # Initialize the DirectoryLoader
    loader = DirectoryLoader(
        '/Users/bharatsavanur/Desktop/projects/personal_git_2/ai_ml/langchain_mongodb_rag_openai/data',
        glob="**/*.txt")
    return loader.load()


documents = load_docs()
text_splitter = CharacterTextSplitter(chunk_size=4096, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
print("No Of Documents", len(documents))


print("Total Cunnks:", len(documents))

#   print(document)
# print(data)
# Define the OpenAI Embedding Model we want to use for the source data
# The embedding model is different from the language generation model

print("Starting Embedding Process")

embeddings = OpenAIEmbeddings(openai_api_key=getOpenAIKey())

# Initialize the VectorStore, and
# vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents(documents, embeddings, collection=collection,
                                                      index_name=index_name)

# vectorstoreSearch = MongoDBAtlasVectorSearch(
#   collection=collection, embedding=OpenAIEmbeddings(disallowed_special=()), index_name=index_name
# )

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    get_mongo_uri(),
    "langchain_demo" + "." + "text_data_1",
    OpenAIEmbeddings(disallowed_special=()),
    index_name=index_name,
)

# perform a similarity search between a query and the ingested documents
docs = vector_search.similarity_search("Madam")
print(" Matches", len(docs))

# results = search("bright star")
# print(results)
