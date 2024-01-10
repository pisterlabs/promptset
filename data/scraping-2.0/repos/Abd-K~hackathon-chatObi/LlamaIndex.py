from langchain import OpenAI
from llama_index import SimpleDirectoryReader, PromptHelper, LLMPredictor, ServiceContext, GPTSimpleVectorIndex
import os

os.environ["OPENAI_API_KEY"] = '<insert open AI key here>'
DATA_DIRECTORY_PATH = './docs/current'
INDEX_PATH = 'index.json'
# MODEL_NAME = "text-davinci-003"
# MODEL_NAME = "text-davinci-002"
# MODEL_NAME = "text-ada-003"
MODEL_NAME = "text-curie-003"

TEMPERATURE = 0
max_input_size = 4096
num_outputs = 256
max_chunk_overlap = 20
chunk_size_limit = 600

def construct_index():
    print("constructing index")
    documents = SimpleDirectoryReader(DATA_DIRECTORY_PATH).load_data()
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME, max_tokens=num_outputs))

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    index.save_to_disk(INDEX_PATH)
    print("index created")
    return index

def loadIndex():
    return GPTSimpleVectorIndex.load_from_disk(INDEX_PATH)

