import streamlit as st
import openai
from ticket_ui import create_ticket_ui, display_formatted_ticket, refine_ticket_ui
from common_ticket4 import load_prompts, chat_with_openai, refine_ticket_logic

# Function to validate API key against OpenAI
def validate_api_key(api_key):
    try:
        openai.api_key = api_key
        # Simple, low-cost request to validate key
        openai.Completion.create(engine="text-davinci-003", prompt="Hello world", max_tokens=5)
        return True
    except Exception as e:
        return False

# Function to show the login page
def show_login_page():
    st.title("Welcome to Product GPT")
    st.title("Login")

    # Radio button for login method selection
    login_method = st.radio("Login Method:", ["Username & Password", "API Key"])

    if login_method == "Username & Password":
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")

        if st.button("Login"):
            if username_input == st.secrets["login_credentials"]["username"] and \
                    password_input == st.secrets["login_credentials"]["password"]:
                st.session_state['authenticated'] = True
                # Redirect to ticket creation page by setting the action
                st.session_state['action'] = 'create_ticket'

    else:
        api_key_input = st.text_input("API Key", type="password")

        if st.button("Login with API Key"):
            if validate_api_key(api_key_input):
                st.session_state['authenticated'] = True
                # Redirect to ticket creation page by setting the action
                st.session_state['action'] = 'create_ticket'

# Main function
def main():
    # If the user is not authenticated, show the login page
    if not st.session_state.get('authenticated', False):
        show_login_page()
    else:
        # Define actions for sidebar after login
        if 'action' not in st.session_state:
            st.session_state['action'] = 'create_ticket'

        if st.session_state['action'] == 'create_ticket':
            ticket_type, user_input, format_selection, create_ticket = create_ticket_ui()

            if create_ticket:
                prompts = load_prompts()
                prompt_text = prompts.get(ticket_type, "")
                if not prompt_text:
                    st.error(f"Could not find a prompt for ticket type: {ticket_type}")
                    return

                prompt = {"role": "user", "content": prompt_text + user_input}
                system_prompt = {"role": "system", "content": "You are an experienced product manager and an expert in writing tickets."}
                st.session_state['conversation_history'] = [system_prompt, prompt]

                gpt_response = chat_with_openai(prompt, st.session_state['conversation_history'])
                display_formatted_ticket(gpt_response, format_selection)

            refine_input, refine_ticket = refine_ticket_ui()
            if refine_ticket:
                updated_ticket = refine_ticket_logic(refine_input, format_selection, st.session_state['conversation_history'])
                display_formatted_ticket(updated_ticket, format_selection)

        # Sidebar only appears after successful login
        with st.sidebar:
            if st.button("Create New Ticket"):
                st.session_state['action'] = 'create_ticket'
            if st.button("Log Out"):
                # Clear the session and show the login page
                st.session_state.clear()
                st.session_state['action'] = 'show_login'
                show_login_page()

        # Execute actions as per the session state
        if st.session_state.get('action') == 'logout':
            st.session_state.clear()
            show_login_page()

if __name__ == "__main__":
    main()
