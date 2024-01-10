from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import tenacity
from openai import RateLimitError

print("Setting up API Key...")
openai.api_key = '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'

@tenacity.retry(
    wait=tenacity.wait_fixed(5),  # Wait for 5 seconds between retries
    stop=tenacity.stop_after_attempt(3),  # Retry up to 3 times
    retry=tenacity.retry_if_exception_type(RateLimitError)  # Retry only on RateLimitError
)
def load_data():
    reader = SimpleDirectoryReader(input_dir=r"C:\Users\Shri Karthik\Desktop\techtrek", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in analyzing student data of their particular University, Assume all input prompts to be with respect to the input data, Don't answer anything apart from educational-related prompt"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

# Initialize the chat engine
chat_engine = load_data().as_chat_engine(chat_mode="condense_question", verbose=True)

while True:
    prompt = input("Your question: ")
    if not prompt:
        break
    # Generate a response
    response = chat_engine.chat(prompt)
    print("Assistant:", response.response)
