
import streamlit as st
import os
import hashlib
import pymongo
import certifi
import openai
import configparser
import ast
config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
# db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if "vta_code" not in st.session_state:
    st.session_state.vta_code = None

if 'vta_key' not in st.session_state:
    st.session_state.vta_key = False

if 'api_key' not in st.session_state:
    st.session_state.api_key = False

if 'codes_key' not in st.session_state:
    st.session_state.codes_key = False

if 'temp_uml' not in st.session_state:
    st.session_state.temp_uml = None

if 'teacher_key' not in st.session_state:
    st.session_state.teacher_key = None

if 'student_key' not in st.session_state:
    st.session_state.student_key = False

if 'student_tabs' not in st.session_state:
    st.session_state.student_tabs = None

if 'prompt_bot_key' not in st.session_state:
    st.session_state.prompt_bot_key = None

if 'admin_key' not in st.session_state:
    st.session_state.admin_key = False

if 'engine_key' not in st.session_state:
    st.session_state.engine_key = config['constants']['cb_engine']

if 'bot_key' not in st.session_state:
    st.session_state.bot_key = config['constants']['cb_bot']

# config ini values change
if 'cb_settings_key' not in st.session_state:
    st.session_state.cb_settings_key = {
        "cb_temperature": float(config['constants']['cb_temperature']),
        "cb_max_tokens": float(config['constants']['cb_max_tokens']),
        "cb_n": float(config['constants']['cb_n']),
        "cb_presence_penalty": float(config['constants']['cb_presence_penalty']),
        "cb_frequency_penalty": float(config['constants']['cb_frequency_penalty'])
    }


def teacher_login():
    with st.form(key="authenticate"):
        st.write("For Teachers")
        teacher_code = st.text_input('Teacher code:')
        teacher_code = teacher_code.lower()
        password = st.text_input('Password:', type="password")
        submit_button = st.form_submit_button(label='Login')
    if submit_button:
        user = user_info_collection.find_one({"tch_code": teacher_code})
        if user:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            if user["pass_key"] == hashed_password:
                st.write("Login successful!")
                vta_codes = []
                for i in range(1, 46):
                    code = user.get("vta_code{}".format(i), None)
                    if code is None:
                        break
                    vta_codes.append(code)
                api_key = user.get("api_key", None)
                st.session_state.bot_key = user.get(
                    "bot_key", st.session_state.bot_key)
                st.session_state.engine_key = user.get(
                    "engine_key", st.session_state.engine_key)
                st.session_state.codes_key = vta_codes
                st.session_state.api_key = api_key
                st.session_state.vta_code = teacher_code
                st.session_state.teacher_key = teacher_code
                st.session_state.vta_key = True
                return True
            else:
                st.error("Incorrect password!")
        else:
            st.error("User not found!")
    return False


def class_login():
    with st.form(key='access'):
        st.write("For students")
        vta_code = st.text_input('VTA code: ')
        vta_code = vta_code.lower()
        submit_button = st.form_submit_button(label='Start')
        if submit_button:
            query_conditions = [
                {"vta_code" + str(i): vta_code} for i in range(1, 46)]
            query = {"$or": query_conditions}
            projection = {"_id": 0, "api_key": 1,
                          "tch_code": 1,  "bot_key": 1, "engine_key": 1}
            cursor = user_info_collection.find(query, projection)
            for doc in cursor:
                if "api_key" in doc:
                    result = doc["api_key"]
                    st.session_state.bot_key = doc.get(
                        "bot_key", st.session_state.bot_key)
                    st.session_state.engine_key = doc.get(
                        "engine_key", st.session_state.engine_key)
                    st.session_state.teacher_key = doc.get("tch_code")
                    st.session_state.vta_key = True
                    st.session_state.vta_code = vta_code
                    st.session_state.api_key = result
                    st.session_state.student_key = True
                    return True
            st.error(
                "VTA code not found, please request for a VTA code before starting your session")
            return False
