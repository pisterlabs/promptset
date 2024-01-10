import json
import psycopg2
from variables_and_prompts import *
import openai




def insert_into_database(user_id, workout_plan):
    # insert a complete workout plan into database
    json_format= workout_plan.split("{")[1]
    json_format = json_format.split("}")[0]
    json_format = "{"+ json_format+ "}"

    json_format = json.loads(json_format)

    # create database connection
    for item in json_format:
        the_date = item
        workout = json_format[the_date]
        execute("INSERT INTO workout_schedule (date, workout, user_id) VALUES (%s, %s,%s);", [the_date, workout, user_id], commit=True)
        
    
def delete_create_history(user_id):
    execute("DELETE FROM generation_chat_history WHERE user_id = %s", [user_id], commit = True)

def delete_specific_date(user_id, date):
    execute("DELETE FROM workout_schedule WHERE user_id = %s AND date = %s", [user_id, date], commit=True)
    
def delete_workout_for_user(user_id):
    execute("DELETE FROM workout_schedule WHERE user_id = %s", [user_id], commit = True)
    
def insert_specific_date(user_id, date):
    pass

def set_goals(chat_history, user_id):
    
    # remove old goals
    
    chat_history_object = execute("SELECT * FROM generation_chat_history WHERE user_id = %s ORDER BY id;", [user_id], commit=False)
    chat_history = []
    content = ""
    for message in chat_history_object:
        if message[3] != "system":
            content += message[2]
        
    chat_history.append({"role": "system", "content": prompt_get_user_information+content})
    execute("DELETE FROM user_information WHERE user_id = %s", [user_id], commit=True)
    print(1)
    chat_history.insert(0,{"role": "system", "content": prompt_get_user_information})
    print(2)
    model = "gpt-3.5-turbo-0613"

    response = openai.ChatCompletion.create(
        model=model,
        messages=chat_history,
        temperature=0,
    )
    
    response = response.choices[0].message["content"]
    print(response)
    json_format= response.split("{")[1]
    json_format = json_format.split("}")[0]
    json_format = "{"+ json_format+ "}"
    json_format = json.loads(json_format)

    # create database connection
    for item in json_format:
        the_thing = item
        value = json_format[the_thing]
        execute("INSERT INTO user_information (goal, user_id) VALUES (%s,%s);", [value, user_id], commit=True)
        
    
    
def get_user_id(username):

    try:
        conn = psycopg2.connect(
            dbname="dblua8qg5ehr18",
            user="brjyyccwesckpy",
            password="638b0040bc3765bf41a90f060604f05e2130fd1daf9382bf72dfa3dd4807f589",
            host="ec2-52-17-31-244.eu-west-1.compute.amazonaws.com",
            port="5432"
        )
        cursor = conn.cursor()
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)
    
    try:
        select_query = "SELECT id FROM credentials WHERE username = %s;"
        cursor.execute(select_query, (username,))
        user_id = cursor.fetchone()[0]

    except (Exception, psycopg2.Error) as error:
        conn.rollback()
        print("Error inserting row:", error)

    return user_id


def get_chat_history(user_id):
    try:
        conn = psycopg2.connect(
            dbname="dblua8qg5ehr18",
            user="brjyyccwesckpy",
            password="638b0040bc3765bf41a90f060604f05e2130fd1daf9382bf72dfa3dd4807f589",
            host="ec2-52-17-31-244.eu-west-1.compute.amazonaws.com",
            port="5432"
        )
        cursor = conn.cursor()
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)

    # Check security on this query
    query = "SELECT * FROM user_chat_history WHERE user_id = %s ORDER BY message_id;"

    cursor.execute(query, (user_id, ))
    
    chat_history_object = cursor.fetchall()

    chat_history = []

    for message in chat_history_object:
        msg_object = {"role": message[2], "content": message[3]}
        chat_history.append(msg_object)

    cursor.close()
    conn.close()

    return chat_history

def update_chat_history(user_id, role, new_message):
    try:
            conn = psycopg2.connect(
                dbname="dblua8qg5ehr18",
                user="brjyyccwesckpy",
                password="638b0040bc3765bf41a90f060604f05e2130fd1daf9382bf72dfa3dd4807f589",
                host="ec2-52-17-31-244.eu-west-1.compute.amazonaws.com",
                port="5432"
            )
            cursor = conn.cursor()
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)

    query = """INSERT INTO user_chat_history (user_id, role, content) VALUES (%s, %s, %s);"""

    cursor.execute(query, (user_id, role, new_message))

    conn.commit()

    cursor.close()
    conn.close()
def execute(query, variables = [], commit = False):
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur = conn.cursor()
        cur.execute(query, variables)
        
        if commit:
            conn.commit()
            results = []
        else: 
            results = cur.fetchall()

        cur.close()
        conn.close()
        return results
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)
    

    
    
    