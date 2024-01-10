import llama_index, os
import dill as pickle # dill is a more powerful version of pickle
from llama_index import ServiceContext, StorageContext
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

load_dotenv('app/.env')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0
)

llm_embeddings = OpenAIEmbeddings()

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=llm_embeddings
)

llama_index.set_global_service_context(service_context)

# The other computational tasks
ccel_storage_context = StorageContext.from_defaults(persist_dir='app/systematic_theology')

# if precomputed_results directory doesn't exist, create it
if not os.path.exists('precomputed_results'):
    os.makedirs('precomputed_results')

# Serialize with dill
with open('precomputed_results/ccel_storage_context.pkl', 'wb') as f:
    pickle.dump(ccel_storage_context, f)