from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, Document, LLMPredictor, LangchainEmbedding

from custom_llm import CustomLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

llm = LLMPredictor(CustomLLM())
model_name = "sentence-transformers/all-mpnet-base-v2"
embed = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))   

index = GPTSimpleVectorIndex.load_from_disk('data.json')
index._include_extra_info = True

response = index.query("query")
print(response)

