from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, Document, LLMPredictor, LangchainEmbedding
from custom_llm import CustomLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

llm = LLMPredictor(CustomLLM())
model_name = "sentence-transformers/all-mpnet-base-v2"
embed = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))   

# Load documents into index
documents = SimpleDirectoryReader(input_files=[
                                  r"C:\Users\data"]).load_data()
index = GPTSimpleVectorIndex.from_documents(documents)

index.save_to_disk('data.json')

