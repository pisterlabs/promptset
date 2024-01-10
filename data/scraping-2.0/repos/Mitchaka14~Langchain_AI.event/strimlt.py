import os
import shutil
import sqlite3
import pandas as pd
import streamlit as st
from langchain import LLMMathChain, SerpAPIWrapper, OpenAI, LLMChain
from langchain.agents import (
    AgentType,
    initialize_agent,
    Tool,
)
from langchain.chat_models import ChatOpenAI
from tools.my_tools import DataTool, SQLAgentTool, EmailTool
import subprocess
import os
from PIL import Image

from dotenv import load_dotenv

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    os.environ["serpapi_api_key"] = st.secrets["SERPAPI_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["serpapi_api_key"] = os.getenv("SERPAPI_API_KEY")
search = SerpAPIWrapper()

data_tool = DataTool()
sql_agent_tool = SQLAgentTool()
email_sender_tool = EmailTool()
sql_agent_tool.description = ""

tools = [
    data_tool,
    sql_agent_tool,
    email_sender_tool,
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to ask with search..use for realtime questions like time etc and some internet related things.....",
    ),
]

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
# _________________________________________________________________________________________
db_path = os.path.join(os.getcwd(), "ClinicDb.db")  # The full path of the database file
recovery_db_path = os.path.join(os.getcwd(), "ClinicDbRecovery.db")


# Function to display and edit business info text file
def business_info():
    file_path = os.path.join("data", "business_info.txt")
    with open(file_path, "r+") as file:
        content = file.read()
        updated_content = st.text_area("Business Info:", content)
        if st.button("Save Changes"):
            file.seek(0)
            file.write(updated_content)
            file.truncate()


def get_table_info(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()


def handle_db_upload(uploaded_file):
    if uploaded_file is not None:
        # Check if a database already exists and remove it
        if os.path.exists(db_path):
            os.remove(db_path)

        # Save the uploaded file as the new database
        with open(db_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            st.success("Uploaded file successfully!")

        # Update SQLAgentTool's description
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            sql_agent_tool.description = (
                "Here are the tables and columns available to use:\n"
            )
            for table_name in tables:
                table_name = table_name[0]
                sql_agent_tool.description += f"\nTable: {table_name}\n"
                columns = get_table_info(cursor, table_name)
                sql_agent_tool.description += "Columns:\n" + "\n".join(
                    [column[1] for column in columns]
                )


def display_and_edit_table(conn, table_name):
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    st.dataframe(df)
    st.subheader(f"Edit entries from {table_name}")
    column_to_edit = st.selectbox(
        "Select column to edit", df.columns, key=f"{table_name}_select"
    )
    if column_to_edit:
        entry_to_edit = st.text_input(
            "Enter the entry to edit", key=f"{table_name}_{column_to_edit}_edit"
        )
        new_value = st.text_input(
            "Enter the new value", key=f"{table_name}_{column_to_edit}_value"
        )
        if st.button(
            f"Update {table_name}", key=f"{table_name}_{column_to_edit}_button"
        ):
            query = f"UPDATE {table_name} SET {column_to_edit} = ? WHERE {column_to_edit} = ?"
            try:
                conn.execute(query, (new_value, entry_to_edit))
                conn.commit()
                st.success("Entry updated successfully!")
            except sqlite3.Error as e:
                st.error(f"An error occurred: {e}")


# Function to display and edit database
def database_info():
    uploaded_file = st.file_uploader("Upload a new database (optional)", type="db")
    handle_db_upload(uploaded_file)

    if st.button("Reset"):
        if os.path.exists(db_path):
            os.remove(db_path)
        subprocess.call(["python", "ClinicDb_create.py"])
        sql_agent_tool.description = ""
        st.success("Database reset successfully!")

    if os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table_name in tables:
                table_name = table_name[0]
                st.subheader(f"Table: {table_name}")
                display_and_edit_table(conn, table_name)


def handle_chat(user_input, system_message):
    # Add user input to chat history
    st.session_state["chat_history"].append(("user", user_input))

    # Format the input to agent.run()
    formatted_input = "\n".join(
        [system_message]
        + [f"{name}: {message}" for name, message in st.session_state["chat_history"]]
    )

    # Run the agent
    output = agent.run(input=formatted_input)

    # Extract agent response from output
    agent_response = output.split("Final Answer:")[-1].strip()

    # Add agent response to chat history
    st.session_state["chat_history"].append(("assistant", agent_response))


def presentation():
    # Import required libraries
    from PIL import Image
    import os

    # Define the path where your images are stored
    images_folder = "./VoiceVerse AgentAI"  # Corrected path

    # Ensure the images are sorted by their names (1.jpg, 2.jpg, ..., 11.jpg)
    images = sorted(
        [
            os.path.join(images_folder, img)
            for img in os.listdir(images_folder)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ],
        key=lambda x: int(
            os.path.splitext(os.path.basename(x))[0]
        ),  # Sort the images by their names (1, 2, ..., 11)
    )

    # Initialize image index in Session State
    if "img_idx" not in st.session_state:
        st.session_state["img_idx"] = 0

    # Open and display the image:
    image = Image.open(images[st.session_state["img_idx"]])
    st.image(image, use_column_width=True)

    # Create two columns for buttons
    col1, col2 = st.columns(2)

    # Add a button to column 1
    if col1.button("Previous Image"):
        # Decrement image index, ensuring it doesn't go below 0
        st.session_state["img_idx"] = max(0, st.session_state["img_idx"] - 1)

    # Add a button to column 2
    if col2.button("Next Image"):
        # Increment image index, ensuring it doesn't go beyond the number of available images
        st.session_state["img_idx"] = min(
            len(images) - 1, st.session_state["img_idx"] + 1
        )


def main():
    st.title("Customer Service App")
    agent_type = st.sidebar.selectbox(
        "Select Agent Type",
        options=[AgentType.OPENAI_FUNCTIONS, AgentType.ZERO_SHOT_REACT_DESCRIPTION],
        format_func=lambda x: x.name,  # Display enum name
    )
    agent = initialize_agent(
        tools, llm, agent=agent_type, verbose=True
    )  # Update agent with selected AgentType
    # Adding navigation
    pages = {
        "Chat": st.empty,
        "Business Info": business_info,
        "Database Info": database_info,
        "Presentation": presentation,
    }
    page = st.sidebar.radio("Navigation", tuple(pages.keys()))

    # Call the function of the selected page
    pages[page]()

    if page == "Chat":
        # Initialize chat history in Session State
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Initialize user details in Session State
        if "user_details" not in st.session_state:
            st.session_state["user_details"] = {"full_name": "", "email": ""}

        # Add input fields for full name and email address
        if not st.session_state["user_details"]["full_name"]:
            st.session_state["user_details"]["full_name"] = st.text_input(
                "Enter your full name:"
            )
        if not st.session_state["user_details"]["email"]:
            st.session_state["user_details"]["email"] = st.text_input(
                "Enter your email address:"
            )

        full_name = st.session_state["user_details"]["full_name"]
        email = st.session_state["user_details"]["email"]

        # Use a chat input for user input
        user_input = st.chat_input("Enter your message here:")

        # Define the system message
        system_message = f"""
        (system: You are a friendly Human named Jack (dont Jack: )who is a customer service agent. and this conversation is with a customer
        before trying to perform any task always ask customers for their full name...You must know their name
          For appointments or anything booking related, query the ClinicDBTool!!  )

        User Full Name: {full_name}
        User Email: {email}
        send Email for reminders and other important things
        ask customer if they want to be sent certain things depending on its nature
        """

        # Check if there's user input
        if user_input:
            handle_chat(user_input, system_message)

        # Display the chat history
        for name, message in st.session_state["chat_history"]:
            with st.chat_message(name):
                st.markdown(message)

        # Button to reset conversation
        if st.button("Reset Conversation"):
            st.session_state["chat_history"] = []
            st.session_state["user_details"] = {"full_name": "", "email": ""}


if __name__ == "__main__":
    main()
