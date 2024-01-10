import streamlit as st
import pinecone
import openai

from apikey import OPENAI_KEY, PINECONE_KEY, PINECONE_ENV

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate

# Set Streamlit page configuration
st.set_page_config(page_title='🧠 Marcus', layout='wide')
# Set up the Streamlit app layout
st.title("🧠 Marcus")

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


TEMPLATE= """Pretend you are a stoic philosopher from Ancient Greece named Marcus.
Return responses in the style
of an ancient Greek philosopher like Epictetus or Seneca. Please cite stoic thinkers and 
their writings if they are relevant to the discussion.
Sign off every response with "Sincerely, Marcus".

User input: {user_input}"""

PROMPT = PromptTemplate(
    input_variables = ['user_input'],
    template = TEMPLATE
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
pinecone.init(
    api_key=PINECONE_KEY,
    environment=PINECONE_ENV
    )
index_name = 'marcus'

docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Marcus:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

API_O = st.sidebar.text_input("API-KEY", type="password")
# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    llm = ChatOpenAI(
        openai_api_key=API_O,
        model_name='gpt-3.5-turbo',
        temperature=0.2)

    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=5,
                return_messages=True)

    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever()
        )

    tools = [
        Tool(
            name='Stoic Compendium',
            func=qa.run,
            description=(
                'use this tool when answering philosophical queries'
            )
        )
    ]
             
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=st.session_state.entity_memory
        )

else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.stop()



# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
user_input = get_text()


# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    prompt_with_query = PROMPT.format(user_input = user_input)
    response = agent(prompt_with_query)
    answer = response["output"]
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(answer)  

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)










