# import
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings import BedrockEmbeddings

# load the document and split it into chunks
loader = PyPDFLoader("Text_Book_for_Year_6_Science_Knowledge.pdf")
documents = loader.load()
# load_and_split

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embeddings = BedrockEmbeddings(
        # credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1" #sets the region name (if not the default)
        # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
    ) #create a Titan Embeddings client
# embedding_function = SentenceTransformerEmbeddings(model_name="ai21.j2-ultra-v1")

# load it into Chroma
db = Chroma.from_documents(docs, embeddings)

# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)