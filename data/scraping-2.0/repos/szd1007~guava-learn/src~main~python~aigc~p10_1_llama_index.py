import openai, os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, GPTSimpleKeywordTableIndex

openai.api_key = os.environ.get("OPEN_API_KEY")

documents = SimpleDirectoryReader('/Users/zm/aigcData/mr_fujino').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

index.storage_context.persist('/Users/zm/aigcData/index_mr_fujino')