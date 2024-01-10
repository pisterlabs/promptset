import os
import openai
from dotenv import dotenv_values
from langchain.chat_models import AzureChatOpenAI
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
import time
import sys
from chatBot import chatBot
sys.path.append("../..")   ## add directory above
from rag_skills.utils import count_tokens

### Cofigurations
VERBOSE = True
TEMPERATURE = 0.0
TOP_P = 1.0
NUM_CHUNKS = 10
MAX_TOKEN_FOR_CONTEXT = 27000
VECTOR_COL_NAME = "Embedding" ## Column name in ACS for vector embedding
CHUNK_NAME = "Chunk" ## Column name in ACS for text data that contains the context

TEMPLATE_QA_CHAIN = """You are a chatbot having a conversation with a human. 
        Given the Context, Chat History, and Human Query, 
        answer without hallucinating. If you don't have the answer say "I don't have the answer" """

TEMPLATE_CONTEXT_SUMMARIZATION =  """
        Summarize the context so it includes the details related to the human query. """

# Get the absolute path to the .env file
env_name = os.path.join(os.path.dirname(__file__), "llm.env")

# Load environment variables from the .env file
config = dotenv_values(env_name)

if len(config) == 0:
    env_name = os.path.join(os.path.dirname(__file__), "../../llm.env")
    config = dotenv_values(env_name)

    if len(config) == 0:
        raise Exception("No environment variables loaded. Please check the *.env file.")

for key, value in config.items():
    os.environ[key] = value

# LOAD OpenAI configs
openai.api_type = config["OPENAI_API_TYPE"]
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = config['OPENAI_API_BASE']
openai.api_version = config['OPENAI_API_VERSION']
print("ENV VARIABLES LOADED")

# Model choice
DEPLOYMENT_NAME = config['OPENAI_DEPLOYMENT_COMPLETION']

## Azure cognitive search
cogsearch_name = os.getenv("COGSEARCH_NAME")
index_name = os.getenv("COGSEARCH_INDEX_NAME")
cogsearch_api_key = os.getenv("COGSEARCH_API_KEY")
service_endpoint = "https://" + config["COGSEARCH_NAME"] + ".search.windows.net"

credential = AzureKeyCredential(cogsearch_api_key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

optional_params = {
    'top_p': TOP_P,
}

llm = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    openai_api_key=openai.api_key,
    openai_api_type = openai.api_type,
    temperature=TEMPERATURE,
    model_kwargs=optional_params,
    verbose=VERBOSE,
)

## Chatbot class that implements the conversation agent

cb = chatBot(
    llm,
    search_client,
    max_token_for_context=MAX_TOKEN_FOR_CONTEXT, 
    template_qa_chain=TEMPLATE_QA_CHAIN,
    template_context_summarization=TEMPLATE_CONTEXT_SUMMARIZATION, 
    numChunks=NUM_CHUNKS,
    vectorColName=VECTOR_COL_NAME,
    chunkColName=CHUNK_NAME,
    to_debug=VERBOSE
)

def get_answer(msg):
    ans = cb.run(msg)
 
    return ans


if __name__ == '__main__':
    question = """what are the top 3 themes in the earnings call transcripts from ticker MSFT for the quarter 1 in year 23?
    """
    
    start = time.time()
    ans = get_answer(question)
    end = time.time()
    print(ans)
    print("Time elapsed: {}".format(end-start))
    
    result_num_tokens = count_tokens(ans)
    print("Response num tokens: {}".format(result_num_tokens))

    ans = get_answer("Is it possible to get more details about cloud ?")
    print(ans)