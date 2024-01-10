# https://techcommunity.microsoft.com/t5/educator-developer-blog/teach-chatgpt-to-answer-questions-using-azure-cognitive-search/ba-p/3969713
# Library imports
from collections import OrderedDict
import requests

# Langchain library imports
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Configuration imports
from config import (
    SEARCH_SERVICE_ENDPOINT,
    SEARCH_SERVICE_KEY,
    SEARCH_SERVICE_API_VERSION,
    SEARCH_SERVICE_INDEX_NAME1,
    SEARCH_SERVICE_SEMANTIC_CONFIG_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
)

# Cognitive Search Service header settings
HEADERS = {
    'Content-Type': 'application/json',
    'api-key': SEARCH_SERVICE_KEY
}

def search_documents(question):
    """Search documents using Azure Cognitive Search"""
    # Construct the Azure Cognitive Search service access URL
    url = (SEARCH_SERVICE_ENDPOINT + 'indexes/' +
               SEARCH_SERVICE_INDEX_NAME1 + '/docs')
    # Create a parameter dictionary
    params = {
        'api-version': SEARCH_SERVICE_API_VERSION,
        'search': question,
        'select': '*',
        '$top': 3,
        'queryLanguage': 'en-us',
        'queryType': 'semantic',
        'semanticConfiguration': SEARCH_SERVICE_SEMANTIC_CONFIG_NAME,
        '$count': 'true',
        'speller': 'lexicon',
        'answers': 'extractive|count-3',
        'captions': 'extractive|highlight-false'
        }
    # Make a GET request to the Azure Cognitive Search service and store the response in a variable
    resp = requests.get(url, headers=HEADERS, params=params)
    # Return the JSON response containing the search results
    return resp.json()
        
def filter_documents(search_results):
    """Filter documents that score above a certain threshold in semantic search"""
    file_content = OrderedDict()
    for result in search_results['value']:
        # The '@search.rerankerScore' range is 0 to 4.00, where a higher score indicates a stronger semantic match.
        if result['@search.rerankerScore'] > 1.5:
            file_content[result['metadata_storage_path']] = {
                'chunks': result['pages'][:10],
                'captions': result['@search.captions'][:10],
                'score': result['@search.rerankerScore'],
                'file_name': result['metadata_storage_name']
            }

    return file_content

def create_embeddings():
    """Create an embedding model"""
    return OpenAIEmbeddings(
        openai_api_type='azure',
        openai_api_key=AZURE_OPENAI_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        deployment='text-embedding-ada-002',
        model='text-embedding-ada-002',
        chunk_size=1
    )

def store_documents(docs, embeddings):
    """Create vector store and store documents in the vector store"""

    # add chroma!!!!!!

    return FAISS.from_documents(docs, embeddings)

def answer_with_langchain(vector_store, question):
    """Search for documents related to your question from the vector store
    and answer question with search result using the lang chain"""

    # add a chat service
    llm = AzureChatOpenAI(
        openai_api_key=AZURE_OPENAI_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        openai_api_type='azure',
        deployment_name='gpt-35-turbo',
        temperature=0.0,
        max_tokens=500
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    return chain({'question': question})

def main():
    QUESTION = 'Tell me about effective prompting strategies'

    # Search for documents with Azure Cognitive Search

    search_results = search_documents(QUESTION)

    file_content = filter_documents(search_results)

    print('Total Documents Found: {}, Top Documents: {}'.format(
        search_results['@odata.count'], len(search_results['value'])))

    
    # 'chunks' is the value that corresponds to the Pages field that you set up in the Cognitive Search service.
    # Find the number of chunks
    docs = []
    for key,value in file_content.items():
        for page in value['chunks']:
            docs.append(Document(page_content = page,
                             metadata={"source": value["file_name"]}))
    print("Number of chunks: ", len(docs))

    # Answer your question using the lang chain

    embeddings = create_embeddings()

    vector_store = store_documents(docs, embeddings)

    result = answer_with_langchain(vector_store, QUESTION)

    print('Question: ', QUESTION)
    print('Answer: ', result['answer'])
    print('Reference: ', result['sources'].replace(",","\n"))

# execute the main function
if __name__ == "__main__":
    main()