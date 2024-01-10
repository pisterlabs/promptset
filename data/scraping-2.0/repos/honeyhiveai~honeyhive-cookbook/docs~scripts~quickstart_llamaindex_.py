pip install honeyhive -q
import honeyhive
import os
from honeyhive.utils.llamaindex_tracer import HoneyHiveLlamaIndexTracer

os.environ["HONEYHIVE_API_KEY"] = "YOUR_HONEYHIVE_API_KEY"

tracer = HoneyHiveLlamaIndexTracer(
    project="PG Q&A Bot",  # necessary field: specify which project within HoneyHive
    name="Paul Graham Q&A",  # optional field: name of the chain/agent you are running
    source="staging",  # optional field: source (to separate production & staging environments)
    user_properties={  # optional field: specify user properties for whom this was ran
        "user_id": "sd8298bxjn0s",
        "user_account": "Acme"                                 
        "user_country": "United States",
        "user_subscriptiontier": "enterprise"
    }
)
from llama_index import VectorStoreIndex, SimpleWebPageReader, ServiceContext
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize the service context with the HoneyHive tracer
callback_manager = CallbackManager([tracer])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["http://paulgraham.com/worked.html"]
)

# Pass the service_context to the index that you will query
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
honeyhive.sessions.feedback(
    session_id = tracer.session_id,
    feedback = {
        "accepted": True,
        "saved": True,
        "regenerated": False,
        "edited": False
    }
)
