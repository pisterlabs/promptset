import os
import openai
import uuid
from typing import List
import mysql.connector
import time
from mysql.connector import connect, MySQLConnection
from mysql.connector.cursor import MySQLCursor

def get_connection(autocommit: bool = True) -> MySQLConnection:
    connection = connect(host='127.0.0.1',
                         port=4000,
                         user='root',
                         password='',
                         database='test')
    connection.autocommit = autocommit
    return connection

def read_file_lines(file_path):
    lines = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return lines

def read_string_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

schema = read_string_from_file("tpch-queries/schema.sql")

querymap = {}
for i in range(1, 19):
    query = read_string_from_file(f"tpch-queries/{i}.sql")
    querymap[i] = query

rewrites = read_file_lines('prompts.txt')

def total_cost(steps) -> float:
    return sum(float(value[2]) for value in steps)

def get_cost(sql) -> float:
    try:
        connection = get_connection(autocommit=True)
        cursor = connection.cursor()
        cursor.execute("explain format=verbose " + sql)
        result = cursor.fetchall()
        return total_cost(result)
    except Exception as error:
        #print(error)
        return -1.0

openai.api_key = os.getenv("OPENAI_API_KEY")
print ("\n ----------------- results -------------------- \n")

for i in range(1, 19):
    if i == 15:
        continue
    sql = querymap[i]
    for rw in rewrites:
        prompt_value="### MySQL tables, with their properties:\n#\n#"+schema+"\n#\n### "+rw+sql+" \n"
        time.sleep(3)
        response = openai.Completion.create(
                model="text-davinci-003",
                #model="text-davinci-002",
                prompt=prompt_value,
                temperature=0,
                max_tokens=256,
                n=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["#", ";"]
        )
        new_sql = (response.choices[0].text).replace('\n', ' ')
        original_cost = get_cost(sql)
        new_cost = get_cost(new_sql)
        if new_cost > 0 and new_cost < original_cost:
            #print("QUERY =", i, " ORIGINAL COST = ", original_cost, " NEW COST = ", new_cost, " PROMPT : ", rw)
            print ("ORIGINAL SQL = ",sql, " WITH COST = ", original_cost, "\n REWRITTEN SQL= ",new_sql, " WITH COST= ", new_cost)
print ("\n ---------------------------------------------- \n")

