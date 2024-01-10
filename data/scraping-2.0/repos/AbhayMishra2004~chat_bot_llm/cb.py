import os
import streamlit as st
import numpy as np
import langchain
from langchain.text_splitter import CharacterTextSplitter
from scipy import spatial
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory


################### Helper Functions ###########################

def get_api_key():
    """Retrieve API key from environment variables."""
    try:
        api_key = ""
    except KeyError:
        print("Error: LL_API_KEY environment variable not found.")
        return None
    return api_key


def update_chat_history(role, message):
    """ Updates the chat history with new message """
    st.session_state.messages.append({"role": role, "content": message})

    with st.chat_message(role):
        st.markdown(message)


def get_embeddings(sentence, model):
    """  Returns embeddings for a given sentence using a specified model """
    return model.encode(sentence)


def get_top_answers(question_embeddings, embeddings, cleaned_texts):
    """ Returns top answers according to cosine similarity """
    cos_sim = [1 - spatial.distance.cosine(question_embeddings, e) for e in embeddings]

    top_2_ans_indices = np.argpartition(cos_sim, -3)[-3:]

    return [cleaned_texts[idx] for idx in top_2_ans_indices]


def chat_model(api_key, temperature, length_penalty):
    # Instantiate Llama API and ChatLlamaAPI models

    ########## METADA OF THE CHATMODEL ###############
    chat_model_metadata = {
        "api_request_json": {
            "model": "llama-70b-chat",
            "stream": False
        },
        "temperature": temperature,
        "length_penalty": length_penalty,
    }

    # api_key = get_api_key()
    llama = LlamaAPI(api_key)
    model = ChatLlamaAPI(client=llama, metadata=chat_model_metadata)
    return model


def chat_bot_response(user_input, api_key, chunk_size, temperature, length_penalty, file=None):
    """ Returns chatbot response for provided user input. """
    page_contents = None
    memory = st.session_state["chat_memory"]

    model = chat_model(api_key, temperature, length_penalty)

    if file:
        # Get filename and extension
        filename, file_extension = os.path.splitext(file)

        # Handle file based on extension
        if file_extension == ".pdf":
            loader = PyPDFLoader(file)
        else:
            loader = TextLoader(file, encoding="UTF-8")

        documents = loader.load()

        text_splitter = CharacterTextSplitter(separator=".", chunk_size=chunk_size, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # DONE EMBEDDING AND CREATED VECTOR SPACE
        db = FAISS.from_documents(docs, embedding_function)

        # Similarity search BETWEEN QUESTION AND DATA BASE
        docs = db.similarity_search(user_input, k=3)

        # Extracting page_content into an array, while replacing '\n' with ' '
        page_contents = [doc.page_content.replace('\n', ' ') for doc in docs]

    prompt = f"Question: {user_input} "

    if page_contents:
        prompt += "for given context \n\nPossible Answers:\n\n"
        for answer in page_contents:
            prompt += f"- {answer}\n"

    chain = ConversationChain(llm=model, memory=memory)

    response = chain.run(user_input)

    # Update session state for next turn
    st.session_state["chat_memory"] = memory

    return response


################### Initializevariables ###########################

st.set_page_config(page_title="VK18 GPT", page_icon="🤖")

st.title('Abhay Chatbot')

# Display the beta testing message
st.markdown("This chatbot is currently in beta testing and is still under development.")
st.markdown("Enter the Llama API key , to run the chatbot.")
st.markdown("You can get your free API key from : https://www.llama-api.com/")

################### DOCUMENT  ###########################

# Initialize session state if needed
if "chat_memory" not in st.session_state:
    st.session_state["chat_memory"] = ConversationBufferMemory()

############## API BOX API ####################
# Display a label and text box to enter the API key
with st.sidebar:
    api_key = text_input = st.text_input(
        "API Key 👇",
        label_visibility="visible",
        disabled=False,
        placeholder="Enter Llama API key",
    )
    print("api_key", api_key)

############ TEAMPRATURE SET #################
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    help="Controls the creativity of the model. Lower values lead to more conservative responses, while higher values lead to more creative responses.",
)

length_penalty = st.sidebar.slider(
    "Length Penalty",
    min_value=0.0,
    max_value=2.0,
    step=0.5,
    value=1.0,
    help="Controls the length of the chatbot's responses. Higher values encourage shorter responses, while lower values allow for longer responses. A value of 0.0 means no length restriction."
)

#################### FILENAME ########################
filename = None

# with st.sidebar:
#     uploaded_file = st.file_uploader("Choose a file")

#     if uploaded_file is not None:
#         filename = uploaded_file.name

############### CHUNK SIZE SET ################

if filename:
    with st.sidebar:
        chunk_size = st.slider(
            label="Chunk Size:",
            min_value=20,
            max_value=500,
            value=200,
            help="Adjusts the size of text chunks used for processing. Smaller chunk sizes may result in more granular control over the chatbot's responses, while larger chunk sizes may improve efficiency and reduce processing time.",
        )
else:
    chunk_size = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask something")

if user_input:

    update_chat_history("user", user_input)

    try:
        response = chat_bot_response(user_input, api_key, chunk_size, temperature, length_penalty, filename)
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
        response = "Sorry, an error occurred"

    update_chat_history("assistant", response)
