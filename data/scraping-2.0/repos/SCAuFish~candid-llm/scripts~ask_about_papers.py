from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from llama_index.llms import OpenAI
import openai


# TODO: Set your OpenAI API key
openai.api_key = ""
# TODO: set path to your folder with pdf files
DATA_FOLDER = "./hallucination_papers_subset"


documents = SimpleDirectoryReader(DATA_FOLDER).load_data()

# set context window
context_window = 4096
# set number of output tokens
num_output = 256
# define LLM
llm = OpenAI(temperature=0.1, model="gpt-4")
service_context = ServiceContext.from_defaults(
    llm=llm,
    context_window=context_window,
    num_output=num_output
)

# build index
index = KeywordTableIndex.from_documents(documents, service_context=service_context)

# get response from query
chat_engine = index.as_chat_engine()

while True:
    print("=========================")
    prompt = input("User: ")
    print("=========================")
    streaming_response = chat_engine.chat(prompt)
    print("Agent: ")
    print(streaming_response)
