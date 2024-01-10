# Import required libraries  
import os  
import json  
import uuid
import openai
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (  
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SearchField,  
    SemanticSearch,
    VectorSearch,  
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch,  
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
) 


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


service_endpoint ="https://gptdocument.search.windows.net"
index_name="gptdocument"
key = "W7XVOLsbEZgcvWqLD0MxpdVbAfWf1ZIZG7HYdIVSgPAzSeDTMtqR"
credential = AzureKeyCredential(key)
def create_azure_search_index():
    
    # client = AzureOpenAI(
    #     api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    #     api_version = "2023-05-15",
    #     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    # )
    
    
    
    # Create a search index
    index_client = SearchIndexClient(
        endpoint=service_endpoint, credential=credential)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        # SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="page", type=SearchFieldDataType.Int32,
                        filterable=True),
        # SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        #             searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    ]

    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            )
        ]
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            # title_field=SemanticField(field_name="title"),
            # keywords_fields=[SemanticField(field_name="category")],
            content_fields=[SemanticField(field_name="content")]
        )
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index with the semantic settings
    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    result = index_client.create_or_update_index(index)
    print(f' {result.name} created')





@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text, model="text-embedding-ada-002"):
    client = AzureOpenAI(
        api_key = "3397748fcdcb4a5fbeb6c2eb5a6a284f",  
        api_version = "2023-05-15",
        azure_endpoint = "https://sean-aoai-gpt4.openai.azure.com/"
    )
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def embedding():
    pdf_loader = PyPDFLoader("jjad179.pdf")
    documents=pdf_loader.load_and_split(CharacterTextSplitter(chunk_size=2000))
    # Generate embeddings for title and content fields
    items=[]
    for item in documents:
        content = item.page_content
        page = item.metadata["page"]
        content_embeddings = generate_embeddings(content)
        items.append({"id":uuid.uuid4().__str__(),"content":item.page_content,"contentVector":content_embeddings,"page":str(page)})
    return items




# create_azure_search_index()
items = embedding()



search_client = SearchClient(service_endpoint, index_name, credential=credential)
search_client.upload_documents(items)
    
# print(f"Uploaded {len(items)} documents in total")  

query = "Introduce the research background"  
  
search_client = SearchClient(service_endpoint, index_name, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="contentVector")
  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["content"]
)  
for result in results:
    print(f"Score: {result['@search.score']}")  
    print(f"Content: {result['content']}")

