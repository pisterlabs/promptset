import os

# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from llama_index.llms.palm import PaLM
from llama_index import StorageContext, load_index_from_storage
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.gpt4all import GPT4AllEmbeddings

embed_model = LangchainEmbedding(GPT4AllEmbeddings())
os.environ['GOOGLE_API_KEY'] = "AIzaSyCiWI65fZsluqcnRwDxM8OXwBeu_zVypBE"
llm = PaLM()


# Configure prompt parameters and initialise helper
# max_input_size = 400
# num_output = 400
# max_chunk_overlap = 0.3

# prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Load documents from the 'data' directory
documents = SimpleDirectoryReader('Data').load_data()
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=8000, chunk_overlap=20, embed_model=embed_model)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir="./storage")