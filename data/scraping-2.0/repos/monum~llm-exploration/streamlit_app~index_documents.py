import os
os.environ['OPENAI_API_KEY']= "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"

from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext
from langchain.chat_models import ChatOpenAI

max_input_size    = 4096
num_outputs       = 100
max_chunk_overlap = 20
chunk_size_limit  = 600
prompt_helper = PromptHelper(max_input_size, 
                             num_outputs, 
                             max_chunk_overlap = max_chunk_overlap, 
                             chunk_size_limit = chunk_size_limit)

llm_predictor = LLMPredictor(
    llm = ChatOpenAI(
        temperature = 0.7,
        model_name = "gpt-3.5-turbo",
        max_tokens = num_outputs
    )
)
documents = SimpleDirectoryReader("data").load_data()
index = GPTVectorStoreIndex.from_documents(
            documents, 
            llm_predictor = llm_predictor, 
            prompt_helper = prompt_helper)

index.storage_context.persist(persist_dir=".")
