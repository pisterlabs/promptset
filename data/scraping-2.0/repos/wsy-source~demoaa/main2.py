from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
import os
from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import SemanticField, SemanticConfiguration,SemanticSearch,SemanticPrioritizedFields
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.messages import HumanMessage
# Import required libraries  
import os  

from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (

    VectorizedQuery,
)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


model="gpt-4-32k"
embeddings_model="text-embedding-ada-002"
llm=AzureChatOpenAI(
    openai_api_type="azure",azure_endpoint="https://sean-aoai-gpt4.openai.azure.com/",
    api_key="3397748fcdcb4a5fbeb6c2eb5a6a284f",api_version="2023-07-01-preview",
    model=model,callbacks=[StreamingStdOutCallbackHandler()],streaming=True)


splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
pdf_content = PyPDFLoader("jjad179.pdf")
content = pdf_content.load_and_split(splitter)

AZURE_SEARCH_ENDPOINT="https://gptdocument.search.windows.net"
ADMIN_KEY="W7XVOLsbEZgcvWqLD0MxpdVbAfWf1ZIZG7HYdIVSgPAzSeDTMtqR"
INDEX_NAME="gptdocument"

prompt = """
Please help me generate a 1500-word Chinese article, divided into sections according to background, purpose, methods, main results, discussion, and conclusion.

Document: {document}

"""
template=PromptTemplate.from_template(prompt)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text, model="text-embedding-ada-002"):
    client = AzureOpenAI(
        api_key = "3397748fcdcb4a5fbeb6c2eb5a6a284f",  
        api_version = "2023-05-15",
        azure_endpoint = "https://sean-aoai-gpt4.openai.azure.com/"
    )
    return client.embeddings.create(input = [text], model=model).data[0].embedding

credential=AzureKeyCredential(ADMIN_KEY)
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, INDEX_NAME, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings("Please help me generate a 1500-word Simplified Chinese article, divided into sections according to background, purpose, methods, main results, discussion, and conclusion."), k_nearest_neighbors=10, fields="contentVector")
  
results= search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["content"]
)

print(results)
chain = LLMChain(prompt=template,llm=llm,verbose=True)
chain.run(document=list(results))