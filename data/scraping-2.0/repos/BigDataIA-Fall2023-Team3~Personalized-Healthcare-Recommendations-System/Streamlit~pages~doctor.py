import streamlit as st
import base64
import psycopg2
import time
import requests
import json
import pandas as pd
import bcrypt
import os
import openai
from pinecone import init, Index
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Parameters for connecting to the database
db_host = st.secrets["DB_HOST"]
db_name = st.secrets["DB_NAME"]
db_user = st.secrets["DB_USER"]
db_password = st.secrets["DB_PASSWORD"]
conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host,
    port=5432  # Default PostgreSQL port
)
cursor = conn.cursor()

pinecone_api_key = st.secrets['PINECONE_API_KEY']
pinecone_env = 'gcp-starter'
init(api_key=pinecone_api_key, environment=pinecone_env)
index = Index('bigdata')

gender_choices = ["Male", "Female", "Other"]
def set_bg_hack(main_bg):
    main_bg_ext = "jpeg"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack(st.secrets["IMAGE_PATH"])

def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password)


# Function to check if session has timed out
def check_session_timeout(session_start_time, timeout_minutes=30):
    current_time = time.time()
    elapsed_time = current_time - session_start_time
    return elapsed_time > (timeout_minutes * 60)
from openai import OpenAI
client = OpenAI()
def perform_pinecone_search(txt):
    embedding = client.embeddings.create(model="text-embedding-ada-002", input=txt).data[0].embedding
    
    res = index.query(vector=embedding, top_k=1, include_metadata=True)

    return res
##############################################################################################################

#App Content
st.title('DOCTORS PORTAL')
with st.expander("### How to Navigate the Doctor Portal"):
    st.write("""
    This Doctor's Portal is designed to streamline patient diagnosis and treatment. 
    - **Login:** Doctors begin by logging into their account using their unique credentials.
    - **Patient Details:** Once logged in, doctors can input patient details including gender, age, and symptoms.
    - **Diagnosis and Treatment:** By submitting these details, the portal utilizes the OpenAI Embed API and Pinecone search to provide potential diagnoses and recommended treatments based on the symptoms entered.
    - **Security:** The portal ensures security through encrypted passwords and session time-outs for added confidentiality.
    - **User-Friendly Interface:** An intuitive design allows for easy navigation, making it an efficient tool for medical professionals.
    """)

            
##############################################################################################################
#Login          
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'display_content' not in st.session_state:
    st.session_state['display_content'] = None

if not st.session_state['logged_in']:
    st.sidebar.subheader("Login to Your Account")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        cursor.execute("SELECT * FROM doctor WHERE username = %s", (username,))
        result = cursor.fetchone()
        if result:
            hashed_password = result[-1]
            if isinstance(hashed_password, str):
                hashed_password = hashed_password.encode('utf-8')
            if hashed_password and check_password(hashed_password, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['session_start_time'] = time.time()
                st.session_state['doctor_details'] = result 
            else:
                 st.warning("Incorrect Password")
        else:
            st.warning("Incorrect Username not found")

##############################################################################################################
#Doctor Details
if st.session_state['logged_in']:
    # Check for session timeout
    if check_session_timeout(st.session_state['session_start_time']):
        st.session_state['logged_in'] = False
        st.warning("Session has timed out. Please login again.")
    else:
       # Show patient details
        doctor_details = st.session_state['doctor_details']
        doctor_id, doctor_f, doctor_l, doctor_p, doctor_e, doctor_username, doctor_password = doctor_details
        with st.expander("Doctor Details"):
            st.write(f"Doctor ID: {doctor_id}")
            st.write(f"First Name: {doctor_f}")
            st.write(f"Last Name: {doctor_l}")
            st.write(f"Practice: {doctor_p}")
            st.write(f"Email: {doctor_e}")
            st.write(f"Username: {doctor_username}")

        Gender = st.selectbox("Select the Gender of the Patient", gender_choices)
        Age = st.number_input("Enter the Age of the Patient", min_value=0, max_value=120)
        Symptoms = st.text_input("Symptoms")
        AI = st.text_input("Optional: Additional Information")

        if st.button("Submit") and Gender != '' and Age != '' and Symptoms != '':
            st.session_state['display_content'] = 'Submit'
        else:
            st.warning("Please fill in all the fields")

        if st.session_state['display_content'] == 'Submit':
            if Symptoms != '' and Age != '' and Gender != '':
                # Perform Pinecone search
                
                st.header("Medical Diagnosis and Treatment Recommendation")
                try:
                    search_results = perform_pinecone_search(Symptoms)
                    if len(search_results['matches']) > 0:
                        metadata = search_results['matches'][0]['metadata']
                        with st.expander("Diagnostic Information"):
                            st.write("Diagnosis:", metadata.get("Diagnosis", ""))
                        with st.expander("Treatment Information"):
                            st.write("Recommended Treatment:", metadata.get("Treatment", ""))
                    else:
                        st.warning("No matching diagnosis and treatment found.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.error("Please try again.")
            else:
                st.warning("Please fill in all the fields")
            





##############################################################################################################
        
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

conn.close()



