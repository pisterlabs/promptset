from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleKeywordTableIndex, RAKEKeywordTableIndex#, KeywordTableIndex
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
import os

def set_service_context(key=None):
    if key is None:
        key = os.environ.get('OPENAI_API_KEY')
    service_context = ServiceContext.from_defaults(llm=OpenAI(api_key=key), embed_model=OpenAIEmbedding(api_key=key))
    return service_context

def get_index(index_name):
    Index = None
    if index_name == 'VectorStoreIndex':
        Index = VectorStoreIndex
    elif index_name == 'SimpleKeywordTableIndex':
        Index = SimpleKeywordTableIndex
    elif index_name == 'RAKEKeywordTableIndex':
        Index = RAKEKeywordTableIndex
    else:
        print('index name not imported')
    return Index
