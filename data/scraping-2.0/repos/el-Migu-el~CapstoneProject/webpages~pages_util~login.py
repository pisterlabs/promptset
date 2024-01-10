import streamlit as st
import hmac
import os
from openai import OpenAI, AuthenticationError
import pandas as pd
from webpages.pages_util.util import CUSTOMER_DATA_PATH


def is_valid_api_key(api_key):
    messages = [{"role": "user", "content": "Hello!"}]

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        return True
    except AuthenticationError as e:
        return False


def login():
    """Returns True if the password is correct, otherwise returns False."""

    def load_user_data():
        """Load user data from a CSV file using pandas."""
        user_data = {}
        credentials = {}

        customer_data = pd.read_csv(CUSTOMER_DATA_PATH)

        for index, row in customer_data.iterrows():
            credentials[row['Username']] = row['Password']
            user_data[row['Username']] = {'Email': row['Email'], 'Username': row['Username'],
                                          'Full Name': row['Full Name'], 'Age': row['Age'],
                                          'Location': row['Location'], 'Bot Preferences': row['Bot Preferences']}

        return credentials, user_data

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Login Form"):
            username = st.text_input("Username", key="username")
            password = st.text_input("Password", type="password", key="password")

            api_key = st.text_input("Enter your GPT API key", type="password")
            os.environ["OPENAI_API_KEY"] = api_key.lstrip('"').rstrip('"')

            if st.form_submit_button("Log in") and username and password and api_key:
                if is_valid_api_key(api_key):
                    password_entered()
                else:
                    st.warning("Invalid API key. Please enter a valid GPT API key.")
            else:
                st.warning("Please enter all credentials.")

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        credentials, user_data = load_user_data()  # because we use st.stop, even if this was run outside of check_password, this would always be re-executed if password was incorrect, so I left it here for easier reading
        if st.session_state["username"] in credentials and hmac.compare_digest(
                st.session_state["password"],
                credentials[st.session_state["username"]],
        ):
            st.session_state["logged_in"] = True
            del st.session_state["password"]  # Don't store the password.

            st.session_state["user_data"] = user_data[st.session_state["username"]]
            st.session_state["logging_in"] = False
            st.rerun()
        else:
            st.session_state["logged_in"] = False
            st.error("😕 User not known or password incorrect")

    login_form()

    go_back = st.button('Go Back')
    if go_back:
        st.session_state["logging_in"] = False
        st.rerun()

    # Return True if the username + password is validated, otherwise False.
    return st.session_state.get("logged_in", False)


def signup():
    """Sign up for an account."""

    def sign_up_form():
        """Display the sign-up form."""
        with st.form("Sign Up Form"):
            email = st.text_input('Email')
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            repeat_password = st.text_input('Please Repeat the Password', type='password')
            full_name = st.text_input('First and Last Name')
            age = st.number_input('Age', min_value=0, max_value=120)
            location = st.text_input('Location')
            bot_preferences = st.text_input('Bot Preferences (e.g. "Talk like a butler")')

            submit_button = st.form_submit_button('Submit')
            if submit_button and email and username and password and repeat_password and full_name and age and location and bot_preferences:
                info_submitted(email, username, password, repeat_password, full_name, age, location,
                               bot_preferences)
            else:
                st.warning("Please enter all details.")

    def info_submitted(email, username, password, repeat_password, full_name, age, location, bot_preferences):
        """Process the submitted information."""
        # Check if all fields are filled and passwords match
        customer_data = pd.read_csv(CUSTOMER_DATA_PATH)
        if password != repeat_password or len(password) < 5:
            st.error('Passwords do not match or are too short. Please try again.')
        elif username in customer_data['Username'].values:
            st.error('Username is already taken. Please choose a different one.')
        elif email in customer_data['Email'].values:
            st.error('Email is already taken. Please choose a different one.')

        else:
            # Insert new row into the customer_data DataFrame
            new_row = {'Full Name': full_name, 'Username': username, 'Email': email, 'Password': password,
                       'Age': age, 'Location': location, 'Favorites': [],
                       'Bot Preferences': bot_preferences}

            customer_data = pd.concat([customer_data, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated DataFrame to the CSV file
            customer_data.to_csv(CUSTOMER_DATA_PATH, index=False)

            st.session_state['signup_successful'] = True
            st.session_state["signing_up"] = False
            st.session_state["logging_in"] = True
            st.rerun()

    sign_up_form()

    go_back = st.button('Go Back')
    if go_back:
        st.session_state["signing_up"] = False
        st.rerun()

    return st.session_state.get("signup_successful", False)


def login_signup():
    st.markdown(
        '<p style="font-size:20px; font-weight:bold;">Please log in or create an account to start using AutoMentor!</p>',
        unsafe_allow_html=True)

    login_btn_placeholder = st.empty()
    signup_btn_placeholder = st.empty()

    login_btn = login_btn_placeholder.button('Login')
    signup_btn = signup_btn_placeholder.button('Sign Up')

    if login_btn:
        st.session_state["logging_in"] = True

    if signup_btn:
        st.session_state["signing_up"] = True

    if st.session_state.get("signup_successful", False):
        st.success('Account successfully created! Please log in.')

    if st.session_state.get("logging_in", False):
        login_btn_placeholder.empty()
        signup_btn_placeholder.empty()
        successful_login = login()
        if not successful_login:
            st.stop()

    if st.session_state.get("signing_up", False):
        login_btn_placeholder.empty()
        signup_btn_placeholder.empty()
        successful_signup = signup()
        if not successful_signup:
            st.stop()

    st.write("---")
    st.caption("© 2024 AutoMentor | All rights reserved")
