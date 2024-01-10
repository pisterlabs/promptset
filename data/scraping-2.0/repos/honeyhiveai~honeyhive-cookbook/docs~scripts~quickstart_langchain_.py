pip install honeyhive -q
import os
from honeyhive.sdk.langchain_tracer import HoneyHiveLangChainTracer

HONEYHIVE_API_KEY = "YOUR_HONEYHIVE_API_KEY"
OPENAI_API_KEY = "YOUR_HONEYHIVE_API_KEY"
SERP_API_KEY = "YOUR_SERP_API_KEY"

honeyhive_tracer = HoneyHiveLangChainTracer(
    project="AI Search Chatbot",     # necessary field: specify which project within HoneyHive
    name="SERP Q&A",                 # optional field: name of the chain/agent you are running
    source="staging",                # optional field: source (to separate production & staging environments)
    user_properties={                # optional field: specify user properties for whom this was ran
        "user_id": "sd8298bxjn0s",
        "user_account": "Acme",                                 
        "user_country": "United States",
        "user_subscriptiontier": "enterprise"
    },
    api_key=HONEYHIVE_API_KEY
)
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, Wikipedia
from langchain.agents import Tool, initialize_agent
from langchain.tools import StructuredTool
from langchain.agents.react.base import DocstoreExplorer
from langchain.callbacks import StdOutCallbackHandler

# Initialise the OpenAI LLM and required callables for our tools
llm = OpenAI(
    temperature=0, openai_api_key=OPENAI_API_KEY
)
search = SerpAPIWrapper(
    serpapi_api_key=SERP_API_KEY
)
llm_math_chain = LLMMathChain.from_llm(llm=llm)
docstore = DocstoreExplorer(Wikipedia())

# Define the tools to be fed to the agent
tools = [
    Tool(
        name="Google",
        func=search.run,
        description="Useful for when you need to answer questions about current events. You should ask targeted questions.",
    ),
    Tool(
        name="Wikipedia",
        func=docstore.search,
        description="Useful for when you need factual information. Ask search terms for Wikipedia",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math.",
    )
]
# Initialise the agent with HoneyHive callback handler
agent = initialize_agent(tools=tools, llm=llm)
agent(
    "Which city is closest to London as the crow flies, Berlin or Munich?",
    callbacks=[honeyhive_tracer]
)
import honeyhive

honeyhive.sessions.feedback(
    session_id = honeyhive_tracer.session_id,
    feedback = {
        "accepted": True,
        "saved": True,
        "regenerated": False,
        "edited": False
    }
)
