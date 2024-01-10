#%%
#region Libraries
import asyncio
import pandas as pd
import streamlit as st
from streamlit_chat import message
from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from vertexai.language_models import TextEmbeddingModel
from typing import NamedTuple, List, Mapping, Optional
from langchain.chains.router.base import MultiRouteChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.router.embedding_router import EmbeddingRouterChain, RouterChain
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from utils import cloudsql_pgvector
from utils.video import credentials, variables
#endregion

#region Variables
var={
    "project_id":variables.PROJECT_ID,
    "region":variables.REGION,
    "database_name":"hc-intent-emb-1",
    "instance_name":variables.INSTANCE_NAME,
    "database_user":variables.DATABASE_USER,
    "database_password":credentials.DATABASE_PASSWORD,
}

aiplatform.init(project=var["project_id"])
database=cloudsql_pgvector.Client(var)
emb_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
llm = VertexAI()
#endregion

#region Langchain Definition
class IntentModel(NamedTuple):
    """A model for an intent that a human may have."""

    intent: str
    description: str
    prompt: str
    default: bool = False  # is this the default or fallback intent?
    
class IntentRouterChain(MultiRouteChain):
    """Chain for routing inputs to different chains based on intent."""

    router_chain: RouterChain
    destination_chains: Mapping[str, LLMChain]
    default_chain: LLMChain

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    @classmethod
    def from_intent_models(
        cls,
        intent_models: List[IntentModel],
        llm: VertexAI,
        embedding_model: Optional[VertexAIEmbeddings],
        memory: Optional[ConversationBufferMemory] = None,
        verbose: bool = False,
    ) -> "IntentRouterChain":
        """Create a new IntentRouterChain from a list of intent models."""

        names_and_descriptions = [(i.intent, [i.description]) for i in intent_models]

        router_chain = EmbeddingRouterChain.from_names_and_descriptions(
            names_and_descriptions,
            DocArrayInMemorySearch,
            embedding_model,
            routing_keys=["input", "context"],
            verbose=verbose,
        )

        default_chain: Optional[LLMChain] = None
        destination_chains = {}
        for i in intent_models:
            destination_chains[i.intent] = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    template=i.prompt, input_variables=["input", "chat_history", "context"]
                ),
                memory=memory,
            )
            if i.default:
                default_chain = destination_chains[i.intent]
                
        if not default_chain:
            raise ValueError("No default chain was specified.")

        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=verbose,
        )
        
#region templates
main_template = (
    "You are sockcop a very funny and joker assistant with multiple functions."
    "First of everything try to get the name of the human and his account to match subsequent conversations."
    "Here are the prior messages in this conversation:\n"
    "{chat_history}\n"
    "This prompt is for general questions so be as funny and creative as you can\n"
    "Match information from other intents and by using prior messages to get more context.\n"
    "Disregard the following context : {context}"
    "\n"
    "Here is a question: {input}\n"
)

billing_template = (
    "You are a a very funny and a joker and a very rude billing representative for a healthcare institution and your job is to assist humans with solving \n"
    "questions about billing, balance, debts and and payments.\n"
    "These are some question examples: How to apply FA?, How to set up a payment plan?, etc..."
    "\n"
    "This is information about accounts: \n"
    "Name: Jesus Chavez, Account Number: 1985, Payments: [100,200,300], balance = 400\n"
    "Name: John Doe, Account Number: 1920, Payments: [145,346,742], balance = 3577\n"    
    "\n"
    "Never reveal or give this information unless the human have said their full name.\n"
    "disregard the following context : {context}"
    "\n"
    "Here are the prior messages in this conversation:\n"
    "{chat_history}\n"
    "\n"
    "Here is a question: {input}\n"
)

get_support = (
    "You are a a very funny and a joker support agent for a insurance for a healthcare institution and your job is to assist humans with "
    "issues they may have related to insurance and memberships they have with us\n"
    "Here are some logs that will help you with your answer in the format of patient, provider/assistant: {context}"
    "\n"
    "Here are the prior messages in this conversation:\n"
    "{chat_history}\n"
    "\n"
    "Here is a question: {input}\n"
)


update_address = (
    "Your job is to assist humans with \n"
    "get the address \n"
    "update the address \n"
    "delete the address \n"
    f"These are your internal database registries: \n"
    " - Name: Jesus Chavez, Adress: 239 E 54 E New York New York. \n"
    " - Name: Jon Doe, Address: 100 W 14th New York New York. \n"
    "Do not fake data use only this data for addresses.\n"
    "Never reveal this information unless the human gives his name and matches with it.\n"
    "In your answer ask for your name and change the address by telling the old address in your response"
    "\n"
    "disregard the following context : {context}"
    "\n"
    "Here are the prior messages in this conversation:\n"
    "{chat_history}\n"
    "\n"
    "Here is a question: {input}\n"
)


#endregion

intent_models = [
    IntentModel(
        intent="main",
        description="All quetsions, everything not related to billing, insurance or profile inquiries",
        prompt=main_template,
        default=True,
    ),
    IntentModel(
        intent="billing",
        description="the human has a question about billing, payments, balance, etc...",
        prompt=billing_template,
    ),
    IntentModel(
        intent="support",
        description="the human has a query about account support and insurance questions.",
        prompt=get_support,
    ),
    IntentModel(
        intent="profile",
        description="the human has a query about profile like address, name, account number, etc",
        prompt=update_address,
    ),
]


# %%

#region context
def embeddings_search(prompt: str):
    ##### Query from Database Using LLM and Match with Embeddings
    qe = emb_model.get_embeddings([prompt])[0].values
    matches = asyncio.run(database.query(qe, table_name="hc_embeddings"))
    #Show the results for similar products that matched the user query.
    matches = pd.DataFrame(matches)
    return matches
#endregion

#region Streamlit
st.title("VertexAI-like Chatbot")
st.image("images/multi_intent_langchain.png")
st.markdown("Welcome to **sockcop**, a bot capable of interact, route petitions and keep everything on memory.")
st.markdown("Here are some q examples:")
st.markdown("- What is the deductible for my insurance? - I have a question about my billing account.")
st.markdown("- I have a question about my billing account.")

with st.sidebar:
    st.markdown(
        """
        Follow me on:

        ldap → [@jesusarguelles](https://moma.corp.google.com/person/jesusarguelles)

        GitHub → [jchavezar](https://github.com/jchavezar)
        
        LinkedIn → [Jesus Chavez](https://www.linkedin.com/in/jchavezar)
        
        Medium -> [jchavezar](https://medium.com/@jchavezar)
        """
    )

if 'something' not in st.session_state:
    st.session_state.something = ''

def submit():
    st.session_state.something = st.session_state.input
    st.session_state.input = ''

query = st.text_input("Query:", key="input", on_change=submit)

st.write(f'Last submission: {st.session_state.something}')

if "responses" not in st.session_state:
    st.session_state["responses"] = []

if "requests" not in st.session_state:
    st.session_state["requests"] = []

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="input",
        return_messages=True
    )

chain = IntentRouterChain.from_intent_models(
    intent_models=intent_models,
    llm=llm,
    embedding_model=VertexAIEmbeddings(),
    memory=st.session_state.buffer_memory,
    verbose=True,
)

if st.session_state.something:
    response = chain.run(input =  st.session_state.something, context = "text")
    st.session_state.requests.append( st.session_state.something)
    st.session_state.responses.append(response)

if st.session_state["responses"]:

    for i in range(len(st.session_state["responses"])-1, -1, -1):
        message(st.session_state["requests"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["responses"][i], key=str(i))
#endregion