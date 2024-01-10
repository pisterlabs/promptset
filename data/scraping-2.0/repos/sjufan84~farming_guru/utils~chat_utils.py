# Importing the necessary libraries
import streamlit as st
import openai
import os
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_to_dict
from streamlit_extras.switch_page_button import switch_page
from dotenv import load_dotenv
load_dotenv()
from streamlit_chat import message



# Get the OpenAI API key and org key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")




def init_chat_session_variables():
    # Initialize session state variables
    session_vars = [
        'farmer_response','seed', 'farming_page',  'chat_messages', 'chat_choice','response', 'history', 'chat_history_dict'
    ]
    default_values = [
        '','Spooky', 'general_chat',  [], '', '', None, {}
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value



# Define a function to initialize the chatbot
# This function will be re-used throughout the app
# And take in the initial message as a parameter as well
# as a chat type (e.g. "recipe" or "foodpedia", etc.
# to distinguish between different chatbots and how they are stored in memory
# Set it to the initial message in the chat history
def initialize_chat(initial_message):
    # Initialize the chatbot with the first message
    history = ChatMessageHistory()
    history.add_ai_message(initial_message)
    st.session_state.history = history
    return st.session_state.history

# We need to define a function to save the chat history as a dictionary
# This will be used to save the chat history to the database and to display the chat history
def save_chat_history_dict():
    # Save the chat history as a dictionary
    chat_history_dict = messages_to_dict(st.session_state.history.messages)
    st.session_state.chat_history_dict = chat_history_dict
    return st.session_state.chat_history_dict


# Now we need to define a function to add messages to the chatbot
# This will take in the message_type (e.g. "user" or "ai")
# and the message itself
# It will then add the message to the chat history
# and return the updated chat history
def add_message_to_chat(message, role):
    # Add the appropriate message to the chat history depending on the role
    if role == "user":
        st.session_state.history.add_user_message(message)
    elif role == "ai":
        st.session_state.history.add_ai_message(message)
    
    return st.session_state.history


# Define a function to submit a  question from the user and get the response from the "farmer"
def get_user_question():
    # Create a text input for the user to ask a question
    user_question = st.text_input("Ask a question about farming")
    return user_question


# Define a function to submit a  question from the user and get the response from the "farmer"
def get_farmer_response(question):
    # Check to see if there is an inventory in the session state
    messages = [
    {
        "role": "system",
        "content": f"You are a knowledgeable and helpful farming assistant who can answer the user's various question {question} about farming.\
            The conversation you have had so far is {st.session_state.history.messages}.\
            Please respond as a friendly farmer to help them with their questions."
    },
    {
            "role": "user",
            "content": f"I have a question about farming.  {question}"
    },
    ]

    # Use the OpenAI API to generate a recipe
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = messages,
            max_tokens=750,
            frequency_penalty=0.5,
            presence_penalty=0.75,
            temperature=1,
            n=1
        )
        st.session_state.response = response
        response = response.choices[0].message.content

    except:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages = messages,
            max_tokens=500,
            frequency_penalty=0.2,
            temperature = 1, 
            n=1, 
            presence_penalty=0.2,
        )
        st.session_state.response = response
        response = response.choices[0].message.content

    return response

# Define the general chat function
def general_chat():
    # Create a header
    st.markdown('''<div style="text-align: center;">
    <h4>Farmer Chat 🚜🌾🧑🏼‍🌾</h4>
    </div>''', unsafe_allow_html=True)
    st.text("")

    # Add 1 to the i in session state so we can create unique widgets
    st.session_state.i += 1

    # Initialize the chat if the length of the chat history is 0
    if len(st.session_state.chat_history_dict) == 0:
        initial_message = "What questions can I answer for you today?"
        initialize_chat(initial_message)
    
        # Display initial farmer message
        message(f"{initial_message}", avatar_style='miniavs', seed = f'{st.session_state.seed}')
    # Create a text area for the user to enter their message
    user_message = st.text_area('Ask any farming related question!', value='', height=150, max_chars=None, key=None)
    # Create a button to submit the user message
    submit_user_follow_up_button = st.button("Submit Your Question", type = 'primary', use_container_width=True)
    # Upon clicking the submit button, we want to add the user's message to the chat history and generate a an answer to their question
    if submit_user_follow_up_button:
        with st.spinner('The farmer is thinking about your question...'):
            add_message_to_chat(message = user_message, role = 'user')
            # Generate the response from the chef
            farmer_response = get_farmer_response(question = user_message)
            # Add the response to the chat history
            add_message_to_chat(message = f'{farmer_response}', role = 'ai')
            # Add the new chat history to the chat history dictionary
            st.session_state.chat_history_dict = save_chat_history_dict()
            # Display the chat history dictionary 
            for chat_message in st.session_state.chat_history_dict:
                if chat_message['type'] == 'human':
                    message(chat_message['data']['content'], avatar_style='initials', seed = 'You', key = f'{st.session_state.i}', is_user = True)
                    st.session_state.i += 1
                elif chat_message['type'] == 'ai':
                    message(chat_message['data']['content'], avatar_style='miniavs', seed = f'{st.session_state.seed}', key = f'{st.session_state.i}')

                    st.session_state.i += 1
                    st.session_state.i += 1

# Initializations
init_chat_session_variables()

# Establish the flow of the app
if st.session_state.farming_page == "general_chat":
    general_chat()

