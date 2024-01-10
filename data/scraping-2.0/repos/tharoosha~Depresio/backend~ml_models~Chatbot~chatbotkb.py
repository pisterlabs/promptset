from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper,StorageContext, load_index_from_storage
from langchain import OpenAI
import sys
import json
import os
import openai
from llama_index import ServiceContext
import backoff

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def create_Index(path):
    max_input = 4096
    tokens = 100
    chuck_size = 1000
    max_chunk_overlap = 0.2

    promptHelper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chuck_size)

    #define LLM 
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-ada-003",max_tokens=tokens))

    #load data
    docs = SimpleDirectoryReader(path).load_data()

    #create vector index

    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,prompt_helper=promptHelper)
    # vectorIndex = GPTVectorStoreIndex.from_documents(docs)
    vectorIndex = GPTVectorStoreIndex.from_documents(docs,service_context=service_context)
    # vectorIndex = GPTVectorStoreIndex(docs)
    # vectorIndex
    # vectorIndex.save_to_disk("vactorIndex.json")
    vectorIndex.storage_context.persist(persist_dir = 'Store')
    return vectorIndex

# create_Index()

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)  # Decorate with backoff
def answerMe(question):
    try:
        # create_Index("/Volumes/Transcend/Development/Depresio/ml_models/Chatbot/Knowledge")

        # get query
        # storage_context = StorageContext.from_defaults(persist_dir = '/usr/src/app/ml_models/Chatbot/Store')
        storage_context = StorageContext.from_defaults(persist_dir = '../backend/ml_models/Chatbot/Store')
        index = load_index_from_storage(storage_context)

        query_engine = index.as_query_engine()
        response = query_engine.query(question)

        # data = response.data
        # data_dict = data.to_dict()
        response = str(response)

        output = {"result": response}

        output_json = json.dumps(output)
        print(output_json)
        # print(output)
        # print(response)
        sys.stdout.flush()
        return response
    except openai.error.OpenAIError as oe:
        print(f"OpenAI error: {oe}")
    except Exception as e:
        error_message = str(e)
        output = {"error2011": error_message}

        output_json = json.dumps(output)
        print(output_json)
        sys.stdout.flush()
        # return error_message


if __name__ == '__main__':
    # Read the input from command line arguments
    input = sys.argv[1]
    
    # Call the main function with the input
    answerMe(input)
    # print(answerMe("i'm sleepy right now"))

    



