import streamlit as st
import requests
from dotenv import load_dotenv
import datetime
import pymysql
import os
load_dotenv()
import pinecone
import openai

# Define the API endpoint URL
URL = 'http://bigdata7245-finalproject.ue.r.appspot.com'

if "login_success" not in st.session_state:
    st.session_state['login_success'] = False

def connect_db():
    try:  
        conn = pymysql.connect(
                host = st.secrets["host"], 
                user = st.secrets["user"],
                password = st.secrets["password"],
                db = st.secrets["db"])
        cursor = conn.cursor()
        return conn,cursor
    except Exception as error:
        print("Failed to connect {}".format(error))
            
## Function to check if the user is still eligible to make api calls 
def check_if_eligible(username):
    try:
        print(username)
        if username == "admin":
            return True
        else:
            _,cursor=connect_db()
            query="SELECT * FROM user_data WHERE username='{}'".format(username)
            cursor.execute(query)
            rows = cursor.fetchall()
            rows=rows[0]
            if rows[6] == None:
                last_request_time=datetime.datetime.now()
                update_last_req_time(username,datetime.datetime.now())
                update_count_for_user(username,0)
            elif rows[6] != None:
                last_request_time = datetime.datetime.strptime(str(rows[6]), '%Y-%m-%d %H:%M:%S')
            time_elapsed=datetime.datetime.now() - last_request_time
            if time_elapsed > datetime.timedelta(hours=1):
                update_count_for_user(username,1)
                return True
            elif time_elapsed < datetime.timedelta(hours=1):
                allowed_count=get_allowed_count(rows[5])
                if rows[7] == allowed_count:
                    return False
                elif int(rows[7]) < allowed_count:
                    update_count_for_user(username,rows[7]+1)
                    return True
    except Exception as e:
        print("check_if_eligible: "+str(e))
        return "failed_insert"
    
## Function to update the last request time for the user
def update_last_req_time(username,timestamp):
    try:
        conn,cursor=connect_db()
        query=f"UPDATE user_data SET user_last_request = '{timestamp}' where username='{username}'"
        cursor.execute(query)
        conn.commit()
        conn.close()
    except Exception as e:
        print("update_last_req_time: "+str(e))
        return "failed_insert"
       
## Function to update the count of api calls for the user
def update_count_for_user(username,count):
    try:
        conn,cursor=connect_db()
        query = f"UPDATE user_data SET user_request_count = '{count}' WHERE username = '{username}'"
        cursor.execute(query)
        conn.commit()
        conn.close()
        return "password_updated"
    except Exception as e:
        print("update_count_for_user: "+str(e))
        return "failed_insert" 
        
def get_allowed_count(tier):
        if tier == 'free':
            return 5
        if tier == 'gold':
            return 10
        if tier == 'platinum':
            return 15


def pinecone_init():
    index_name = 'bigdata7245'

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=st.secrets["pinecone_api_key"],
        environment=st.secrets['pinecone_environment'] # find next to api key in console
    )
    # check if 'openai' index already exists (only create index if not)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeds[0]))
    # connect to index
    index = pinecone.Index(index_name)
    return index
    
def query_pinecone_db(query,resturant_id):
    index=pinecone_init()
    openai.api_key = st.secrets['open_api_key']
    MODEL=st.secrets['Embedding_Model']
    xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
    res = index.query([xq], top_k=10, include_metadata=True,namespace=resturant_id)
    #return res['matches']
    reviews=[]
    for match in res['matches']:
        reviews.append(f"{match['score']:.2f}: {match['metadata']['text']}")
    return reviews
    
def chat_gpt(query,prompt):
    openai.api_key = st.secrets['open_api_key']
    response_summary =  openai.ChatCompletion.create(
        model = "gpt-3.5-turbo", 
        messages = [
            {"role" : "user", "content" : f'{query} {prompt}'}
        ],
        temperature=0
    )
    return response_summary['choices'][0]['message']['content']

def query_pinecone(query,resturant_id,username):
    eligible_status=check_if_eligible(username)
    if eligible_status:
        return query_pinecone_db(query,resturant_id)
    else:
        return False

def query_gpt(query, prompt, username):
    eligible_status=check_if_eligible(username)
    if eligible_status:
        return chat_gpt(query, prompt)
    else:
        return False

def handleQuery(_query):

    headers = {"Authorization": "Bearer " + st.session_state['login_token']}

    res = requests.get(URL + '/get_restaurant_id', headers=headers)

    restaurant_id = ""
    if res.status_code == 200:
        restaurant_id = res.json()
        pinecone_res = query_pinecone(_query, restaurant_id, st.session_state['logged_in_username'])
        if pinecone_res:
            for res in pinecone_res:
                st.write(res)
        else:
            st.write("Limit exceeded for 1 hour")

    else: 
        print(res.status_code, res.json())

def analyseReviewsPage():
    st.title('Analyze Reviews')

    with st.form('Analyze'):

        #Query
        _query = st.text_input('Query')

        _submit = st.form_submit_button('Submit')

        if _submit:
            handleQuery(_query)

    with st.form('GPT Query'):
        #GPT Query
        _gpt_query = st.text_input('GPT Query')

        _gpt_prompt = st.text_input('GPT Prompt')

        _gpt_submit = st.form_submit_button('Submit')

        if _gpt_submit:
            _gpt_res = query_gpt(_gpt_query, _gpt_prompt, st.session_state['logged_in_username'])

            if _gpt_res:
                st.write(_gpt_res)
            else:
                st.write("Limit exceeded for 1 hour!")


if __name__ == "__main__":
    if st.session_state.login_success:
        analyseReviewsPage()
    else:
        st.write('Login to access this page!')