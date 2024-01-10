import openai
import pinecone
import streamlit as st
from streamlit_chat import message

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

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
PINECONE_API_KEY = "PLEASE_ENTER_YOUR_OWNED_PINECONE_API_KEY"
PINECONE_ENV = "PLEASE_ENTER_YOUR_OWNED_PINECONE_ENV_NAME"
PINECONE_INDEX_NAME = "PLEASE_ENTER_YOUR_OWNED_PINECONE_INDEX_NAME"

# Set web page title and icon.
st.set_page_config(
    page_title="Chatbot with PDF",
    page_icon=":robot:"
)

# Set web page title and markdown.
st.title('Chatbot with PDF')
st.markdown(
    """
    This is the demonstration of a chatbot with PDF with Azure OpenAI, Pinecone, and Streamlit.
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

# Initialize Pinecone.
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# Initialize session state to store user input.
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Initialize session state to store the chatbot-generated output.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# Initialize Pinecone index and embeddings.
embed = OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME, 
                                    openai_api_key=OPENAI_API_KEY, 
                                    model=OPENAI_EMBEDDING_MODEL_NAME, 
                                    openai_api_type=OPENAI_API_TYPE, 
                                    chunk_size=1)
docsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embed)
user_input = get_input_text()

# Initialize the similarity search.
docs = docsearch.similarity_search(user_input)

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
        },"parameters": {"repetition_penalty": 1.33}
    },
    docs=docs,
    chain=chain)
    
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')