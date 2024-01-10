from llama_index import ServiceContext, SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index.callbacks import OpenAIFineTuningHandler
from llama_index.callbacks import CallbackManager
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is not None:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    # Handle the absence of the environment variable
    # You might want to log an error, raise an exception, or provide a default value
    # For example, setting a default value
    os.environ["OPENAI_API_KEY"] = "your_default_api_key"

data_path = "./test/regression/regression_test003"

documents = SimpleDirectoryReader(
    data_path
).load_data()


finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4", temperature=0.3),
    context_window=2048,  # limit the context window artifically to test refine process
    callback_manager=callback_manager,
)

questions = []
with open(f'{data_path}/generated_data/train_questions.txt', "r") as f:
    for line in f:
        questions.append(line.strip())

from llama_index import VectorStoreIndex

try:
    index = VectorStoreIndex.from_documents(
        documents, service_context=gpt_4_context
    )
    query_engine = index.as_query_engine(similarity_top_k=2)
    for question in questions:
        response = query_engine.query(question)
except Exception as e:
    # Handle the exception here, you might want to log the error or take appropriate action
    print(f"An error occurred: {e}")
finally:
    finetuning_handler.save_finetuning_events(f'{data_path}/generated_data/finetuning_events.jsonl')
