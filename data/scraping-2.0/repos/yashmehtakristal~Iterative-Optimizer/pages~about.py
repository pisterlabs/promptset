import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title = "Iterative Optimizer", page_icon = "🧠", layout = "wide")
st.header("🧠 Iterative Optimizer")

# Add the logo to the sidebar
add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")

import openai
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY


if 'username' in st.session_state:
    st.session_state.username = st.session_state.username

# st.session_state.username = st.session_state.username

def change_states():
    st.session_state.logged_out = True
    st.session_state.logged_in = False
    st.session_state.password_match = None

# let User see app if logged in = True & logged out = False
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    # logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.sidebar.button("Logout", on_click = change_states)

    # Display Markdown of the main page
    st.markdown(
    '''
    Description
    '''
    )

else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")
