from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, PromptHelper
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import OpenAI
import os
import pypdf
import docx2txt


def construct_index(content_data, openai_key):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 1500
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600
    chunk_overlap_ratio = 0.5
    
    os.environ["OPENAI_API_KEY"] = openai_key

    # define LLM or Language Model Wrapper
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs)
    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    service_context = ServiceContext.from_defaults(callback_manager=callback_manager, llm=llm)
    
    documents = SimpleDirectoryReader(content_data).load_data()

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    return query_engine

