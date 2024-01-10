# %%
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

if True:
    import sys

    sys.path.append("../")
from utils.open_search_vector_search_cs import OpenSearchVectorSearchCS
import os

# ENV vars
# empty

# Load Document
doc = PDFMinerLoader('contract.pdf').load()

# Split in chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
split_docs = splitter.split_documents(doc)[:5]

# Add metadata
for doc in split_docs:
    doc.metadata['session_id'] = "123456"

# Create index from documents
embedding_function = OpenAIEmbeddings()

# %%

host = os.getenv('OPENSEARCH_HOST')
port = os.getenv('OPENSEARCH_PORT')
user = os.getenv('OPENSEARCH_USER')
pwd = os.getenv('OPENSEARCH_PWD')

vs = OpenSearchVectorSearchCS(
    opensearch_url=f"https://{host}:{port}",
    embedding_function=embedding_function,
    http_auth=(user, pwd),
    index_name='python-test',
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

filter = {
            "bool" : {
                "filter" : {
                    "term" : {
                        "metadata.session_id" : "123456" 
                    }
                }
            }
        }


search = vs.similarity_search_with_score(
    query="What is the title of the document?", 
    kwargs={
        "search_type":"script_scoring",
        "pre_filter": filter
    }
)
print('Search as vector store client result: ', len(search))


retriever = vs.as_retriever(
    search_kwargs={
        'search_type' : 'script_scoring',
        'pre_filter' : filter
    }
)

rel_docs = retriever.get_relevant_documents(
    query='What is the title of this document?'
)
print('Search as retriever result: ', len(rel_docs))