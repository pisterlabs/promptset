"""
This is the main interface to display 
chat interactions
"""

import openai
import os
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit import components
from streamlit_chat import message
from dotenv import load_dotenv

# Load the services and utils
from utils.chat_utils import ChatService, Context
from utils.cocktail_functions import RecipeService

# Load the environment variables
load_dotenv()


# Set up the OpenAI API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

# Define the page config
st.set_page_config(page_title="BarKeepAI", page_icon="./resources/cocktail_icon.png", initial_sidebar_state="collapsed")

st.markdown("#### Documentation notes:")
st.success('''
           **This is the main chat page.  This can be either a general chat or a recipe chat that takes the generated
           recipe into context when answering questions.**
              ''')
st.markdown('---')


# Define a function to reset the other pages to their default state
def reset_pages():
    st.session_state.cocktail_page = 'get_cocktail_info'
    st.session_state.inventory_page = 'upload_inventory'
    st.session_state.menu_page = 'upload_menus'

reset_pages()


def init_chat_session_variables():
    # Initialize session state variables
    session_vars = [
        'recipe', 'bar_chat_page', 'context', 'i', 'chat_service', 'recipe_service'
    ]
    default_values = [
        None, 'chat_choice', None, 0, ChatService(), RecipeService()
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_chat_session_variables()



def app():
    if st.session_state.bar_chat_page == "chat_choice":
        get_chat_choice()
    elif st.session_state.bar_chat_page == "display_chat":
        display_chat()

def get_chat_choice():
    chat_service = st.session_state.chat_service
    recipe_service = st.session_state.recipe_service
    if recipe_service.recipe:
        st.markdown('**You have created a recipe.  Would you like to ask questions about it, or continue to a general bartender chat?**')
        continue_recipe_button = st.button("Continue with Recipe")
        general_chat_button = st.button("General Chat")
        
        if continue_recipe_button:
            context = Context.RECIPE
            st.session_state.context = context
            chat_service.initialize_chat(context=context)
            st.session_state.bar_chat_page = 'display_chat'
            st.experimental_rerun()
        elif general_chat_button:
            context = Context.GENERAL_CHAT
            st.session_state.context = context
            chat_service.initialize_chat(context=context)
            st.session_state.bar_chat_page = 'display_chat'
            st.experimental_rerun()
    

    else:
        context = Context.GENERAL_CHAT
        st.session_state.context = context
        chat_service.initialize_chat(context=context)
        st.session_state.bar_chat_page = 'display_chat'
        st.experimental_rerun()

def display_chat():
    chat_service = st.session_state.chat_service
    recipe_service = st.session_state.recipe_service
    recipe = recipe_service.recipe
    chat_history = chat_service.chat_history
    
    if chat_history and len(chat_history) == 1:
        initial_prompt = f"What questions can I answer about the {recipe.name}?" if st.session_state.context == Context.RECIPE else "What questions can I answer for you?"
        message(initial_prompt, is_user=False, avatar_style = 'miniavs', seed='Spooky')

    chat_container = st.container()
    with chat_container:
        # Display the chat history
        for i, chat_message in enumerate(chat_history[1:]):
            if chat_message['role'] == 'user':
                message(chat_message['content'], is_user=True, key = f'user_message_{i}')
            elif chat_message['role'] == 'assistant':
                message(chat_message['content'], is_user=False, key = f'ai_message_{i}', avatar_style = 'miniavs', seed='Spooky') 
            else:
                pass
    user_input = st.text_input("Type your message here", key='user_input')
    submit_button = st.button("Submit", type='primary', key='submit_button', use_container_width=True)

    if submit_button:

        with st.spinner("Thinking..."):
            chat_service.get_bartender_response(question=user_input)
            user_input = ""
            st.experimental_rerun()

    st.markdown("---")
    # Create a button to clear the chat history
    clear_chat_history_button = st.button("Clear Chat History", type = 'primary', use_container_width=True)
    # Upon clicking the clear chat history button, we want to reset the chat history and chat history dictionary
    if clear_chat_history_button:
        # Reset the chat history and chat history dictionary
        chat_service.chat_history = []
        chat_service.initialize_chat(context=st.session_state.context)
        # Return to the chat home page
        st.session_state.bar_chat_page = 'display_chat'
        st.experimental_rerun()
    # Create a button to allow the user to create a new recipe
    create_new_recipe_button = st.button("Create a New Recipe", type = 'primary', use_container_width=True)
    # Upon clicking the create new recipe button, we want to reset the chat history and chat history dictionary
    # And return to the recipe creation page
    if create_new_recipe_button:
        # Reset the chat history and chat history dictionary
        chat_service.chat_history = []
        # Return to the recipe creation page
        st.session_state.bar_chat_page = 'get_cocktail_type'
        switch_page("Create Cocktails")
        st.experimental_rerun()


    # Create a button to allow the user to return to "Chat Home"
    return_to_chat_home_button = st.button("Return to Chat Home", type = 'primary', use_container_width=True)
    # Upon clicking the return to chat home button, we want to reset the chat history and chat history dictionary
    # And return to the chat home page
    if return_to_chat_home_button:
        # Reset the chat history and chat history dictionary
        chat_service.chat_history = []
        # Return to the chat home page
        st.session_state.bar_chat_page = 'chat_choice'
        st.experimental_rerun()
                

    # Embed a Google Form to collect feedback
    st.markdown('---')
    st.markdown('''<div style="text-align: center;">
    <h4 class="feedback">We want to hear from you!  Please help us grow by taking a quick second to fill out the form below and to stay in touch about future developments.  Thank you!</h4>
    </div>''', unsafe_allow_html=True)

    src="https://docs.google.com/forms/d/e/1FAIpQLSc0IHrNZvMfzqUeSfrJxqINBVWxE5ZaF4a30UiLbNFdVn1-RA/viewform?embedded=true"
    components.v1.iframe(src, height=600, scrolling=True)



    

app()




