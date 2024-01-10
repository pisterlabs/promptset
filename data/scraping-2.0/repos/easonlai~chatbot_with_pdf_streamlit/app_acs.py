import openai
import streamlit as st
from streamlit_chat import message

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential

# Configure the baseline configuration of the OpenAI library for Azure OpenAI Service.
OPENAI_API_KEY = "PLEASE_ENTER_YOUR_OWNED_AOAI_SERVICE_KEY"
OPENAI_API_BASE = "https://PLESAE_ENTER_YOUR_OWNED_AOAI_RESOURCE_NAME.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_GPT35TURBO_MODEL_NAME"
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_EMBEDDING_MODEL_NAME"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_TYPE = "azure"
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
openai.api_type = OPENAI_API_TYPE
AZURE_COGNITIVE_SEARCH_ENDPOINT_NAME = "https://PLESAE_ENTER_YOUR_OWNED_ACS_RESOURCE_NAME.search.windows.net"
AZURE_COGNITIVE_SEARCH_INDEX_NAME = "PLEASE_ENTER_YOUR_OWNED_ACS_INDEX_NAME"
AZURE_COGNITIVE_SEARCH_KEY = "PLEASE_ENTER_YOUR_OWNED_ACS_SERVICE_KEY"
acs_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_KEY)

# Set web page title and icon.
st.set_page_config(
    page_title="Chatbot with PDF",
    page_icon=":robot:"
)

# Set web page title and markdown.
st.title('Chatbot with PDF')
st.markdown(
    """
    This is the demonstration of a chatbot with PDF with Azure OpenAI, Azure Cognitive Search, and Streamlit.
    I read the book Machine Learning Yearning by Andrew Ng. Please ask me any questions about this book.
    """
)

# Define a function to get user input.
def get_input_text():
    input_text = st.text_input("You: ","Hello!", key="input")
    return input_text 

# Define a function to inquire about the data in Pinecone.
def query(payload, docs, chain):
    response = chain.run(input_documents=docs, question=payload)
    thisdict = {
        "generated_text": response
    }
    return thisdict

# Initialize session state to store user input.
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Initialize session state to store the chatbot-generated output.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# Initialize Azure Cognitive Search index and embeddings.
embed = OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME, 
                                    openai_api_key=OPENAI_API_KEY, 
                                    model=OPENAI_EMBEDDING_MODEL_NAME, 
                                    openai_api_type=OPENAI_API_TYPE, 
                                    chunk_size=1)
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT_NAME,
    azure_search_key=AZURE_COGNITIVE_SEARCH_KEY,
    index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
    embedding_function=embed.embed_query,
)
user_input = get_input_text()

# Initialize the similarity search.
docs = vector_store.similarity_search(user_input)

# Initialize the Azure OpenAI ChatGPT model.
llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME, 
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_base=OPENAI_API_BASE, 
                      openai_api_version=OPENAI_API_VERSION, 
                      openai_api_type = OPENAI_API_TYPE, 
                      temperature=0)

# Initialize the question answering chain.
chain = load_qa_chain(llm, chain_type="stuff")

# Generate the chatbot response.
if user_input:
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        },"parameters": {"repetition_penalty": 1.33} # The repetition penalty is meant to avoid sentences that repeat themselves without anything really interesting.
    },
    docs=docs,
    chain=chain)
    
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')