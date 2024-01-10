import streamlit as st
import sqlite3
import pandas as pd
import os
import tempfile
#from langchain.embeddings.openai import OpenAIEmbeddings
import configparser
import ast

class ConfigHandler:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def get_config_values(self, section, key):
        value = self.config.get(section, key)
        try:
            # Try converting the string value to a Python data structure
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # If not a data structure, return the plain string
            return value
		
config_handler = ConfigHandler()
TCH = config_handler.get_config_values('constants', 'TCH')
STU = config_handler.get_config_values('constants', 'STU')
SA = config_handler.get_config_values('constants', 'SA')
AD = config_handler.get_config_values('constants', 'AD')


# Create or check for the 'database' directory in the current working directory
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]

def fetch_files_with_usernames():
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()
    
    query = '''
    SELECT 
        Files.file_id, 
        Subject.subject_name,
        Topic.topic_name, 
        Files.file_name, 
        Users.username, 
        Files.sharing_enabled
    FROM Files
    JOIN Users ON Files.user_id = Users.user_id
    LEFT JOIN Subject ON Files.subject = Subject.id
    LEFT JOIN Topic ON Files.topic = Topic.id;
    '''
    
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data

def display_files():
    data = fetch_files_with_usernames()
    df = pd.DataFrame(data, columns=["file_id", "subject_name", "topic_name", "file_name", "username", "sharing_enabled"])

    # Convert the 'sharing_enabled' values
    df["sharing_enabled"] = df["sharing_enabled"].apply(lambda x: '✔' if x == 1 else '')

    st.dataframe(
        df, 
        use_container_width=True,
        column_order=["file_id", "subject_name", "topic_name", "file_name", "username", "sharing_enabled"]
    )

def get_file_extension(file_name):
	return os.path.splitext(file_name)[1]

def save_file_to_db(org_id , user_id, file_name, file_content, metadata, subject, topic, sharing_enabled):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()
    extension = get_file_extension(file_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file_content)
        temp_file.flush()

    # Check if the subject already exists. If not, insert it.
    cursor.execute('SELECT id FROM Subject WHERE subject_name = ?', (subject,))
    subject_id_result = cursor.fetchone()
    if not subject_id_result:
        cursor.execute('INSERT INTO Subject (org_id, subject_name) VALUES (?, ?)', (org_id, subject))
        subject_id = cursor.lastrowid
    else:
        subject_id = subject_id_result[0]

    # Check if the topic already exists. If not, insert it.
    cursor.execute('SELECT id FROM Topic WHERE topic_name = ?', (topic,))
    topic_id_result = cursor.fetchone()
    if not topic_id_result:
        cursor.execute('INSERT INTO Topic (org_id, topic_name) VALUES (?, ?)', (org_id, topic))
        topic_id = cursor.lastrowid
    else:
        topic_id = topic_id_result[0]

    # Insert the file data and metadata into the Files table
    cursor.execute(
        'INSERT INTO Files (user_id, file_name, data, metadata, subject, topic, sharing_enabled) VALUES (?, ?, ?, ?, ?, ?, ?);', 
        (user_id, file_name, temp_file.name,metadata, subject_id, topic_id, int(sharing_enabled))
    )
    conn.commit()
    conn.close()




def fetch_subjects_by_org(org_id):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    # Check if the user is a super_admin (org_id is 0)
    if org_id == 0:
        cursor.execute('SELECT * FROM Subject;')
    else:
        cursor.execute('SELECT * FROM Subject WHERE org_id = ?;', (org_id,))
    
    subjects = cursor.fetchall()
    conn.close()
    return subjects


def fetch_topics_by_org(org_id):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    # Check if the user is a super_admin (org_id is 0)
    if org_id == 0:
        cursor.execute('SELECT * FROM Topic;')
    else:
        cursor.execute('SELECT * FROM Topic WHERE org_id = ?;', (org_id,))
    
    topics = cursor.fetchall()
    conn.close()
    return topics

def select_organization():
    with sqlite3.connect(WORKING_DATABASE) as conn:
        cursor = conn.cursor()

        # Org selection
        org_query = "SELECT org_name FROM Organizations"
        cursor.execute(org_query)
        orgs = cursor.fetchall()
        org_names = [org[0] for org in orgs]

        # Use a Streamlit selectbox to choose an organization
        selected_org_name = st.selectbox("Select an organization:", org_names)

        # Retrieve the org_id for the selected organization
        cursor.execute('SELECT org_id FROM Organizations WHERE org_name = ?;', (selected_org_name,))
        result = cursor.fetchone()

        if result:
            org_id = result[0]
            st.write(f"The org_id for {selected_org_name} is {org_id}.")
            return org_id
        else:
            st.write(f"Organization '{selected_org_name}' not found in the database.")
            return None


def docs_uploader():
    st.subheader("Upload Files to build your knowledge base")
    if st.session_state.user['profile_id'] == SA:
        org_id = select_organization()
        if org_id is None:
            return
    else:
        org_id = st.session_state.user["org_id"]
    
    # Upload the file using Streamlit
    uploaded_file = st.file_uploader("Choose a file", type=['docx', 'txt', 'pdf'])
    meta = st.text_input("Please enter your document source (Default is MOE):", max_chars=20)

    # Fetch all available subjects
    subjects = fetch_subjects_by_org(st.session_state.user["org_id"])
    subject_names = [sub[2] for sub in subjects]  # Assuming index 2 holds the subject_name
    selected_subject = st.selectbox("Select an existing subject or type a new one:", options=subject_names + ['New Subject'])
    
    if selected_subject == 'New Subject':
        new_subject = st.text_input("Please enter the new subject name:", max_chars=30)
    else:
        new_subject = None

    # Fetch all available topics
    topics = fetch_topics_by_org(st.session_state.user["org_id"])
    topic_names = [topic[2] for topic in topics]  # Assuming index 2 holds the topic_name
    selected_topic = st.selectbox("Select an existing topic or type a new one:", options=topic_names + ['New Topic'])
    
    if selected_topic == 'New Topic':
        new_topic = st.text_input("Please enter the new topic name:", max_chars=30)
    else:
        new_topic = None
    
    share_resource = st.checkbox("Share this resource", value=True)

    if uploaded_file:
        st.write("File:", uploaded_file.name, "uploaded!")
        if not meta:
            meta = "MOE"

        # Save to Database Button
        if st.button("Save to Database"):
            save_file_to_db(
                org_id=org_id,
                user_id=st.session_state.user["id"],
                file_name=uploaded_file.name,
                file_content=uploaded_file.read(),
                metadata=meta,
                subject=selected_subject if not new_subject else new_subject,
                topic=selected_topic if not new_topic else new_topic,
                sharing_enabled=share_resource
            )
            st.success("File saved to database!")

def fetch_files_by_user_id(user_id):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()
    
    # Fetch files based on user_id
    cursor.execute('SELECT file_name FROM Files WHERE user_id = ?;', (user_id,))
    
    files = cursor.fetchall()
    conn.close()
    return files

def delete_files():
    st.subheader("Delete Files in Database:")
    user_files = fetch_files_by_user_id(st.session_state.user["id"])
    if user_files:
        file_names = [file[0] for file in user_files]
        selected_files = st.multiselect("Select files to delete:", options=file_names)
        confirm_delete = st.checkbox("I understand that this action cannot be undone.", value=False)
        
        if st.button("Delete"):
            if confirm_delete and selected_files:
                delete_files_from_db(selected_files, st.session_state.user["id"], st.session_state.user["profile_id"])
                st.success(f"Deleted {len(selected_files)} files.")
            else:
                st.warning("Please confirm the deletion action.")
    else:
        st.write("No files found in the database.")




def delete_files_from_db(file_names, user_id, profile):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    if profile in [SA, AD]:
        # Delete files irrespective of the user_id associated with them
        for file_name in file_names:
            cursor.execute('DELETE FROM Files WHERE file_name=?;', (file_name,))
    else:
        for file_name in file_names:
            cursor.execute('DELETE FROM Files WHERE file_name=? AND user_id=?;', (file_name, user_id))
            
            # Check if the row was affected
            if cursor.rowcount == 0:
                st.error(f"Unable to delete file '{file_name}' that is not owned by you.")
                
    conn.commit()  # Commit the changes
    conn.close()  # Close the connection


