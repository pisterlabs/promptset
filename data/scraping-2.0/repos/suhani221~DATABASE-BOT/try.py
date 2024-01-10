import openai 
import sqlite3
from sqlite3 import Error
import streamlit as st

openai.api_key = st.secrets.api_credentials.api_key

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('users_db.sqlite')  # connect to the database named users_db.sqlite
    except Error as e:
        st.error(f"Error: {e}")
    return conn

def get_user_by_id(conn, user_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM Users WHERE User_ID = ?", (user_id,))
    user = cur.fetchone()
    return user

def get_all_user_ids(conn):
    cur = conn.cursor()
    cur.execute("SELECT User_ID FROM Users")
    user_ids = cur.fetchall()
    return [id[0] for id in user_ids]

def chat_with_openai(user_id):
    # Create a database connection
    conn = create_connection()

    user_context = None
    if conn is not None:
        user = get_user_by_id(conn, user_id)
        if user is not None:
            user_context = f"A user named {user[1]} who is {user[2]} years old and weighs {user[3]} kilograms. They are experiencing symptoms like {user[4]} and have an allergy to {user[5]}. They live in {user[6]} and work as a {user[7]}. Their menu plan includes {user[8]} and their fitness plan includes {user[9]}."

    # Check if "messages" key is in session_state, if not, initialize it
    if "messages" not in st.session_state:
        st.session_state["messages"] = {}

    if user_id not in st.session_state["messages"]:
        st.session_state["messages"][user_id] = [{"role": "assistant", "content": user_context}]

    for msg in st.session_state["messages"][user_id]:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_context := st.chat_input():
        # openai.api_key = openai_api_key
        st.session_state["messages"][user_id].append({"role": "user", "content": user_context})
        st.chat_message("user").write(user_context)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state["messages"][user_id])
        msg = response.choices[0].message
        st.session_state["messages"][user_id].append(msg)
        st.chat_message("assistant").write(msg.content)

def main():
    # Create a database connection
    conn = create_connection()

    if conn is not None:
        user_ids = get_all_user_ids(conn)
        selected_user_id = st.selectbox('Select a user ID', user_ids)

        # Every time the selected user ID changes, this line gets run again
        chat_with_openai(selected_user_id)

if __name__ == '__main__':
    main()
