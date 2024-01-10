import os
import openai
import uuid
from typing import List
import mysql.connector
import time
import ctypes
import json
import sys

from mysql.connector import connect, MySQLConnection
from mysql.connector.cursor import MySQLCursor
from collections import defaultdict

def get_connection(autocommit: bool = True) -> MySQLConnection:
    connection = connect(host='127.0.0.1',
                         port=4000,
                         user='root',
                         password='',
                         database='test')
    connection.autocommit = autocommit
    return connection

def read_prompts(file_path):
    prompt_dict = defaultdict(list)
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return prompt_dict

def read_string_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

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


def applicable_rewrites(rewrites, query_markers):
    prompts = []
    marker_key_set = set(query_markers)
    for prompt in rewrites["prompts"]:
        if prompt["enabled"].lower() == 'true':
            andTrue = prompt["type"].lower() == 'and'
            if andTrue:
               operators_in_query = marker_key_set.intersection(set(prompt["operators"]))
               if operators_in_query == set(prompt["operators"]):
                   prompts.append(prompt["prompt"])
            else:
               for operator in prompt["operators"]:
                   if operator in marker_key_set:
                       prompts.append(prompt["prompt"])
                       break;
    return set(prompts)

def apply_rewrite(sql, rw):
    skip_gpt = True
    #prompt_value="Assuming the following MySQL tables, with their properties:\n#\n#"+schema+"\n#\n### "+rw+sql+" \n"
    prompt_value=rw+sql
    time.sleep(3)
    message=[{"role": "user", "content": prompt_value}]
    if skip_gpt is True:
        return
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",messages=message,temperature=0,max_tokens=1000
    )
    print ("\n ---------------------------------------------- \n")
    print ("response=",response)
    print ("\n ---------------------------------------------- \n")
    #TODO: fix code below to extract new SQL if any
    new_sql = (response.choices[0].message)
    original_cost = get_cost(sql)
    new_cost = get_cost(new_sql)
    if new_cost > 0 and new_cost < original_cost:
        print ("ORIGINAL SQL = ",sql, " WITH COST = ", original_cost, "\n REWRITTEN SQL= ",new_sql, " WITH COST= ", new_cost)

def apply_rewrites():
    n = len(sys.argv)
    if (n != 2):
        print ("usage: python3 tuner_gpt4.py <test-directory>")
        return
    test_dir = sys.argv[1]
    rewrites = read_prompts('prompts.txt')

    query_dir = test_dir+"/queries"

    filelist = open(test_dir+"/filelist", 'r')
    for onefile in filelist.readlines():
        onefile = onefile.replace('\n','')
        query_file = query_dir+"/"+onefile+".sql"
        f = os.path.join(query_dir, query_file)
        #if not os.path.isfile(f):
           #continue
        sql = read_string_from_file(query_file)
        sql = sql.replace('\n',' ')
        lib = ctypes.CDLL('./analyze.so')
        lib.analyze.restype = ctypes.c_char_p
        key_string = lib.analyze(sql.encode("utf-8"))
        result_file = test_dir+"/results/"+onefile+".out"
        std_file = test_dir+"/std/"+onefile+".out"
        result_handle = open(result_file, "w")
        result_handle.write(key_string.decode("utf-8")+"\n")
        keys = json.loads(key_string.decode("utf-8"))
        for rw in applicable_rewrites(rewrites,keys):
            result_handle.write("rewrite: " + rw+"\n")
            apply_rewrite(sql, rw)

        result_handle.close()
        print("\n diff for ",onefile, "\n")
        os.system("diff "+result_file+" "+std_file)
        print("\n end of diff for ",onefile, "\n")
apply_rewrites()

