import os
import openai
import uuid
from typing import List
import mysql
import time
import ctypes
import json
import sys
import time
import argparse

openai.api_type = "azure"
openai.api_base = "https://zhoushuai-test.openai.azure.com/"
openai.api_version = "2023-07-01-preview"

from mysql.connector import connect, MySQLConnection
from mysql.connector.cursor import MySQLCursor
from collections import defaultdict

def get_connection(autocommit: bool = True) -> MySQLConnection:
    connection = connect(host='127.0.0.1',
                         port=4000,
                         user='root',
                         password='',
                         database='tuner_db')
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

def compare_results(orig_sql, new_sql):
    try:
        connection = get_connection(autocommit=True)
        cursor = connection.cursor()

        # Measure original sql time.
        start_time = time.time()
        cursor.execute(orig_sql)
        orig_result = cursor.fetchall()
        end_time = time.time()
        orig_exec_time = end_time - start_time

        # Measure transformed sql time.
        start_time = time.time()
        cursor.execute(new_sql)
        new_result = cursor.fetchall()
        end_time = time.time()
        new_exec_time = end_time - start_time

        if (orig_result == new_result):
            print(f"Results are same with original_sql exec time = {orig_exec_time} : new_sql exec time = {new_exec_time}")
        else:
            print("Results are different")
            random_filename = os.path.join("/tmp/", str(uuid.uuid4()) + ".json")
            with open(random_filename, "w") as json_file:
                json.dump({"original result": orig_result, "new result": new_result}, json_file)
            print(f"Results dumped to {random_filename}")
    except mysql.connector.Error as e:
        print(f"MySQL error: {e}")
    finally:
        connection.close()

def get_cost(sql) -> float:
    try:
        connection = get_connection(autocommit=True)
        cursor = connection.cursor()
        cursor.execute("explain format=verbose " + sql)
        result = cursor.fetchall()
        return total_cost(result)
    except mysql.connector.Error as e:
        print(f"MySQL error: {e}")
        return -1.0
    finally:
        connection.close()


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

def apply_rewrite(sql, rw, skip_openAI,verify):
    prompt_value=rw+sql
    time.sleep(3)
    message=[{"role": "user", "content": prompt_value}]
    if skip_openAI is True:
        return
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(engine="test-gpt-4",
                                            messages = message,
                                            temperature=0.7,
                                            max_tokens=800,
                                            top_p=0.95,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            stop=None
   )

    new_sql = (response.choices[0].message.content)
    if new_sql == "NO_OPTIMIZATION":
        print("Already optimized sql so no optimization is done")
        return
    print("original sql = ", sql)
    print("new_sql = ", new_sql)
    original_cost = get_cost(sql)
    new_cost = get_cost(new_sql)
    print (" original cost = ", original_cost, "new_cost =", new_cost)
    if (verify):
        compare_results(sql, new_sql)

def tune_one_query(query_file, rewrites, verify, skip_openAI):
        sql = read_string_from_file(query_file)
        sql = sql.replace('\n',' ')
        lib = ctypes.CDLL('./analyze.so')
        lib.analyze.restype = ctypes.c_char_p
        key_string = lib.analyze(sql.encode("utf-8"), False)
        keys = json.loads(key_string.decode("utf-8"))
        applicable_rewrites_list = list(applicable_rewrites(rewrites,keys))
        applicable_rewrites_list.sort()
        for rw in applicable_rewrites_list:
            apply_rewrite(sql, rw, skip_openAI, verify)

def apply_rewrites(test_dir, rewrites, verify, skip_open_ai):

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
        key_string = lib.analyze(sql.encode("utf-8"), False)
        result_file = test_dir+"/results/"+onefile+".out"
        std_file = test_dir+"/std/"+onefile+".out"
        result_handle = open(result_file, "w")
        result_handle.write(key_string.decode("utf-8")+"\n")
        keys = json.loads(key_string.decode("utf-8"))
        applicable_rewrites_list = list(applicable_rewrites(rewrites,keys))
        applicable_rewrites_list.sort()
        for rw in applicable_rewrites_list:
            result_handle.write("rewrite: " + rw+"\n")
            apply_rewrite(sql, rw, skip_open_ai, verify)

        result_handle.close()
        print("\n diff for ",onefile, "\n")
        os.system("diff "+result_file+" "+std_file)
        print("\n end of diff for ",onefile, "\n")


def main():

    parser = argparse.ArgumentParser(description='Database tuner using LLM for optimizing queries.')

    parser.add_argument('test_dir',  help='test queries directory')

    parser.add_argument('--singletest', action='store_true', help='runs single test')
    parser.add_argument('--verify', action='store_true', help='verifies results of the optimized query')
    parser.add_argument('--skip', action='store_true', help='skip optimization and just do rewrite/prompt matching')

    args = parser.parse_args()

    # Access parsed arguments
    verify = args.verify
    singletest = args.singletest
    test_dir = args.test_dir
    skip = args.skip
    rewrites = read_prompts('prompts.txt')

    if singletest:
        tune_one_query(sys.argv[1], rewrites, verify, skip)
    elif test_dir != "":
        apply_rewrites(sys.argv[1], rewrites, verify, skip)
    else:
        print ("usage: python3 tuner.py <test-directory> or \n")
        print ("usage: python3 tuner.py test_file --singletest")


if __name__ == '__main__':
    main()
