import streamlit as st 
import sqlite3
import tensorflow as tf
import numpy as np 
import os 
import csv 
import pandas as pd 
import hashlib
from PIL import Image
from streamlit_option_menu import option_menu
import openai
import re

st.set_page_config(page_title = "PulmoVision")

conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT, state TEXT)')

def make_hash(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

# THIS WILL ADD THE USERNAME AND PASSWORD TO THE DATABASE 
def add_userdata(username,password,state):
    username = str.lower(username)
    if username == "" :
        st.warning("Please Enter a Valid Username")
    else : 
        c.execute("SELECT username FROM userstable") 
        names = {name[0] for name in c.fetchall()} 
        if username in names:  
            st.warning(f"The Username '{username}' is Taken", icon="⛔")
        else : 
            c.execute('INSERT INTO userstable(username,password,state) VALUES (?,?,?)',(username,password,state))
            conn.commit()
            create_patient_table(username)
            st.success("You Have Successfully Created a New Account ✅")

# CHECKS THE TYPED USERNAME AND PASSWORD WITH THE DATABASE 
def login_user(username,password):
    username = str.lower(username)
    if username is None : 
        return False
    state="active"
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ? AND state = ?',(username,password,state,))
    data = c.fetchall()
    if data : 
        return True
    else : 
        return False

# THIS WILL CREATE A PATIENT TABLE WHEN A USER SIGN UP TO THE WEBSITE
def create_patient_table(username):
    table_name = username+'_table'
    tb_create = f"CREATE TABLE IF NOT EXISTS [{table_name}](patient_file_no TEXT,patient_fname TEXT, gender TEXT, patient_age TEXT, diagnosis TEXT, file_name TEXT)"
    c.execute(tb_create)

# VIEW THE ALL PATIENT RECORDS OF THE LOGGED IN DOCTOR
def view_all_patients(username): 
    table_name = username+'_table'
    c.execute(f'SELECT * FROM [{table_name}]')
    data = c.fetchall()
    return data

# ADD PATIENT RECORD TO DOCTOR'S TABLE
def add_patient(username, patient_file_no ,fname,gender,age, diagnosis, file_name):
    table_name = username+'_table'
    c.execute(f'INSERT INTO [{table_name}](patient_file_no,patient_fname,gender,patient_age, diagnosis, file_name) VALUES (?,?,?,?,?,?)',(patient_file_no,fname,gender,age,diagnosis, file_name))
    conn.commit()
    st.success('A New Patient Record is Added ✅')

# MODEL FUNCTION
def mutliclass_model_page(username): 
  model = tf.keras.models.load_model("GP-Multiclass.h5")
  lab = ['COVID-19','NO-FINDINGS','PNEUMONIA','TUBERCULOSIS']

  def processed_img(img_path) : 
      name = os.path.basename(img_path)
      img = tf.keras.utils.load_img(img_path, target_size=(300, 300, 3))
      img_array = tf.keras.utils.img_to_array(img)
      img_array = tf.expand_dims(img_array, 0) 
      predictions = model.predict(img_array)
      score = predictions[0]
      res = "This patient most likely has {}     with a {:.2f} percent confidence.".format(lab[np.argmax(score)], 100 * np.max(score))
      res_csv = format(lab[np.argmax(score)])
      with open("Metadata\metadata.csv", mode='a', newline="") as csvfile : 
          fieldnames = ["file_name", "target"]
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writerow({"file_name":name, "target":res_csv})
      return res

  def run():
      st.markdown('''<h2 style='text-align: center; color: white;'>Decision Support System</h2>''', unsafe_allow_html=True)
      st.subheader("Detect Pneumonia, COVID-19, Tuberculosis in Radiographs")
      st.markdown('''<h5 style='text-align: left; color: white;'>Disclaimer: This tool only used for supporting decision making. Do not use this tool for final diagnosis</h5>''', unsafe_allow_html=True)
      
      patient_file_no = st.text_input("Enter Patient's File Number:", placeholder="e.g. 152468 (Must be 6 or less digits)")
      patient_fname = st.text_input("Enter Patient's Name:", placeholder="Patient's Full Name (Must contain only letters)")
      gender = st.selectbox("Gender:", ["Male", "Female"])
      patient_age = st.slider("Enter patient's age:", 0,120, 25)

      img_file = st.file_uploader("Please, Upload an Image of a chest radiograph", type=["jpg","jpeg", "png"])
      if img_file is not None:
          st.image(img_file,use_column_width=True)
          save_image_path = 'Metadata\\upload_images\\'+img_file.name
          with open(save_image_path, "wb") as f:
              f.write(img_file.getbuffer())

          if st.button("Get Diagnosis"):
            if len(patient_file_no) <= 6 and re.search("[0-9]", patient_file_no) and not patient_file_no.isalpha()\
            and re.match("^[a-z A-Z]*$", patient_fname):
                result = processed_img(save_image_path)
                st.success(result)
                diagnosis = (result[28:42])
                file_name = img_file.name
                add_patient(username, patient_file_no, patient_fname, gender, patient_age, diagnosis, file_name)      
            else : 
                st.warning("File number must contain only 6 or less digits ⚠️")   
                st.warning("Patient name must be valid and no more than 30 characters ⚠️")   

  run()

# admin function to delete accounts
def Delete_account(username) : 
    table_name = username+'_table'
    username_lower = str.lower(username)
    if username_lower == 'admin' :
        st.warning("You Can't Delete The Admin Account Using This Page")
    elif username_lower : 
        c.execute('SELECT * FROM userstable WHERE username=?', (username_lower,))
        data = c.fetchall()
        if data : 
            c.execute(f"DROP TABLE '{table_name}'")
            c.execute("DELETE FROM userstable WHERE username=?", (username_lower,))
            conn.commit()
            st.success("Account Have been Deleted")
        else : 
            st.warning("This Account Doesn't Exist")


# Delete patient function 
def delete_patient(username, patient_number) :
    username = str.lower(username)
    table_name = username+'_table'
    c.execute(f'SELECT * FROM {table_name} WHERE patient_file_no=?', (patient_number,))
    data = c.fetchall()
    if data : 
        c.execute(f"DELETE FROM {table_name} WHERE patient_file_no=?", (patient_number,))
        conn.commit()
        st.success("Patient Record Have Been Sucessfully Deleted ✅")
    else : 
        st.warning("This Patient Record Does Not Exist ⛔")

# Search a single patient function 
def search_patient(username, patient_number): 
    table_name = username+'_table'
    c.execute(f'SELECT * FROM {table_name} WHERE patient_file_no=?',(patient_number,))
    data = c.fetchall()
    if data : 
        patient_record = pd.DataFrame(data, columns=["Patient's File Number","Name", "Gender", "Age", "Diagnoses", "File Name"])
        st.dataframe(patient_record, use_container_width=True)
        csv = convert_df(patient_record)
        st.download_button(label="Export Patient Record as CSV", data=csv,file_name='patient_record.csv',mime='text/csv')
    else : 
        st.warning("This Patient Record Does Not Exist ⛔")

# UPDATE PASSWORD FUNCTION
def update_password(username, password):
    username = str.lower(username)
    if len(password) > 6 and len(password) < 25 and re.search("[^A-Za-z0-9 ]", password) and re.search("[A-Z]", password) : 
        hashed_pass = make_hash(password)
        c.execute('SELECT * FROM userstable WHERE username=?', (username,))
        data = c.fetchall()
        if data : 
            c.execute("UPDATE userstable SET password=? WHERE username=?", (hashed_pass,username,))
            conn.commit()
            st.success("You Have Updated Your Password Successfully ✅")
        else : 
            st.warning("Something Went Wrong ⛔")
    else : 
        st.warning("Please follow the password guidelines")


# convert dataframe to csv file
def convert_df(df):
    return df.to_csv().encode('utf-8')

# view the logged in user patients
def view_patients_table(username) : 
    user_result = view_all_patients(username)
    patient_table = pd.DataFrame(user_result, columns=["Patient's File Number","Name", "Gender", "Age", "Diagnoses", "File Name"])
    st.dataframe(patient_table, use_container_width=True)
    csv = convert_df(patient_table)
    export_button = st.download_button(label="Export as CSV", data=csv,file_name='patient_df.csv',mime='text/csv')

# View all inactive database users by admin
def view_all_inactive_users(): 
    state="inactive"
    c.execute('SELECT username FROM userstable WHERE state=?',(state,))
    data = c.fetchall()
    users_table = pd.DataFrame(data, columns=["Username"])
    st.dataframe(users_table, use_container_width=True)

# View all active database users by admin
def view_all_active_users(): 
    state="active"
    c.execute('SELECT username FROM userstable WHERE state=?',(state,))
    data = c.fetchall()
    users_table = pd.DataFrame(data, columns=["Username"])
    st.dataframe(users_table, use_container_width=True)

# Change account state by admin 
def change_state(username) : 
    c.execute("SELECT username FROM userstable") 
    names = {name[0] for name in c.fetchall()} 
    if username in names:  
        state="active"
        c.execute("UPDATE userstable SET state=? WHERE username=?", (state,username,))
        conn.commit()
        st.success("Username has been activated")
    else : 
        st.warning("This username does not exist")




# Chatbot page function 
def chatbot() : 
    from Constants import OpenAI_API_Key
    openai.api_key = f'{OpenAI_API_Key}'

    st.markdown("<h2 style='text-align: center; color: white;'>The AI Assistant 🧑🏻‍💻</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: white;'>In this page, you can ask a chatbot any question that comes to mind. The\
                model used is based on OpenAI's ChatGPT 💬</h5>", unsafe_allow_html=True)
    placholder_response_user_input = st.empty()
    user_input = placholder_response_user_input.text_input("Enter your question here:", key="user_input", placeholder="e.g. What is Tuberculosis?")
    completion_text = ''
    placeholder_response = st.empty()

    if user_input:
        placeholder_response.text("Waiting for response...")
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt = user_input,
            max_tokens=150,
            temperature =0,
            stream=True,
        )
        for r in response : 
            r_text = r['choices'][0]['text']
            completion_text += r_text
            placeholder_response.markdown(completion_text)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()

def show_main_page(username):
    with mainSection:
        create_usertable()
        username = str.lower(username)
        st.success("Logged In as {}".format(username))
        task = option_menu(
            menu_title=None,
            options=["Profile", "DSS", "Patients", "ChatBot", "Admin"],
            icons=['house-door', 'diagram-3', 'file-medical', 'chat-text', 'person'],
            default_index=2,
            orientation='horizontal')
        if task == "Profile" : 
            st.markdown("<h2 style='text-align: center; color: white;'>My Profile 🙎🏻‍♂️</h2>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: left; color: white;'>Password Update</h4>", unsafe_allow_html=True)
            st.markdown("- Password must be over 6 characters and no more than 25.")
            st.markdown("- Password must contain atleast one capital letter and one special character.")                
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
            password = st.text_input("New Password", placeholder="Enter your new password here", type="password")
            confirmed_pass = st.text_input("Confirm New Password", placeholder="Enter your new password again", type="password")
            if st.button("Update Password") : 
                if password == confirmed_pass : 
                    update_password(username, confirmed_pass)
                else : 
                    st.warning("The Passwords Entered Do Not Match ⚠️")

        elif task == "DSS" : 
            mutliclass_model_page(username)

        elif task == "Patients" : 
            patient_option = option_menu(
            menu_title=None,
            options=["View Patients", "Delete Patients", "Search Patients"],
            icons=["caret-down-square","caret-down-square","caret-down-square"],
            default_index=0,
            orientation='horizontal')

            # view all patients
            if patient_option == 'View Patients' : 
                view_patients_table(username)
            # Delete patient
            elif patient_option == 'Delete Patients' :
                patient_number = st.text_input("Enter Patient File Number to be Deleted:", placeholder="e.g. 82742")
                delete_patient_btn = st.button("Delete Patient Record")
                if delete_patient_btn : 
                    delete_patient(username, patient_number)
            # Search through patients records
            elif patient_option == 'Search Patients' : 
                patient_number = st.text_input("Enter Patient File Number:", placeholder="e.g. 82742")
                search_patient_btn = st.button("Retrieve Patient Record")
                if search_patient_btn : 
                    search_patient(username, patient_number)
            # st.markdown("<h2 style='text-align: center; color: white;'>View All of Your Patients Records 📋</h2>", unsafe_allow_html=True)
        elif task == "ChatBot" :
            chatbot()
        elif task == "Admin" : 
            if username == 'admin': 
                st.markdown("<h2 style='text-align: center; color: white;'>Users Management Tab</h2>", unsafe_allow_html=True)
                admin_option = option_menu(
                menu_title=None,
                options=["View Users", "Delete User"],
                icons=["caret-down-square","caret-down-square"],
                default_index=0,
                orientation='horizontal')
                if admin_option == 'View Users' : 
                    user_account_state = st.radio("Users' Account State:",
                    ('Active', 'Inactive'))
                    if user_account_state == 'Active':
                        view_all_active_users()
                    elif user_account_state == 'Inactive' : 
                        view_all_inactive_users()
                        activate_username = st.text_input("Enter username to activate:", placeholder="username")
                        activate_btn = st.button("Activate Account")
                        if activate_btn : 
                            change_state(activate_username)

                elif admin_option == 'Delete User' : 
                    st.markdown("From this page you can terminate users accounts")
                    st.markdown("Disclaimer: All stored information linked with the entered username will be removed aswell")
                    deleted_user = st.text_input("User's username", placeholder="Write the username you wish to terminate")
                    terminate_button = st.button("Terminate Account")
                    if terminate_button : 
                        Delete_account(deleted_user)
            else : 
                st.warning("You Do Not Have Permission To Access This Page")


def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False

def show_logout_button():
    loginSection.empty()
    with logOutSection:
        st.button ("Log Out", key="logout", on_click=LoggedOut_Clicked)

def LoggedIn_Clicked(username, password):
    # check with database
    if login_user(username, password):
        st.session_state['loggedIn'] = True
        st.session_state['username'] = username
    else:
        st.session_state['loggedIn'] = False
        st.warning("Invalid Credentials or an Inactive Account ⛔")

def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            sign_op = option_menu(
            menu_title=None,
            options=["Log-In", "Sign-Up"],
            icons=['box-arrow-in-right', 'door-open'],
            default_index=0,
            orientation='horizontal')
            if sign_op == 'Log-In' : 
                username = st.text_input (label="Username:", value="", placeholder="Enter your username")
                password = st.text_input (label="Password:", value="", placeholder="Enter your password", type="password")
                hashed_password = make_hash(password)
                st.button("Login", on_click=LoggedIn_Clicked, args= (username, hashed_password)) 
                return username
            elif sign_op == 'Sign-Up' : 
                st.write("📋 Creating a username and password guidelines:")
                st.markdown("- Username must be less than 8 characters. You can use '.' and '-' in your username.")
                st.markdown("- Password must be over 6 characters and no more than 25.")
                st.markdown("- Password must contain atleast one capital letter and one special character.")                
                st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)

                new_user = st.text_input("Username:", placeholder="Enter your username")
                new_password = st.text_input("Password:", type='password', placeholder="Enter your password")
                if st.button("Signup") :
                    if len(new_user) <= 8 and all(c.isalpha() or c == '-' or c == '.' for c in new_user)\
                    and len(new_password) > 6 and len(new_password) < 25 and re.search("[^A-Za-z0-9 ]", new_password) and re.search("[A-Z]", new_password) :
                        hashed_password = make_hash(new_password)
                        account_state = "inactive"
                        create_usertable()
                        add_userdata(new_user, hashed_password, account_state) 
                    else : 
                        st.warning("Please follow the provided guidelines for creating an account")


with headerSection:
    col1, col2, col3 = st.columns(3)
    image = Image.open('logo.png')
    with col1:
        st.write('')
    with col2:
        st.image(image, use_column_width=True)
    with col3:
        st.write('')
    st.markdown("<h1 style='text-align: center; color: white;'>PulmoVision</h1>", unsafe_allow_html=True)
    #first run will have nothing in session_state
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        show_login_page() 
    else:
        if st.session_state['loggedIn']:
            show_logout_button() 
            username = st.session_state.get("username", "")
            show_main_page(username)
        else:
            show_login_page()
