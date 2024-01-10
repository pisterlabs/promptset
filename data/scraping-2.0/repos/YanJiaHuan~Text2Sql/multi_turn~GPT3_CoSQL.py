import pandas as pd
import time
import openai
import os
import sys
import tiktoken
import sqlite3
#################### 0. Prompt   ####################
SQL_generation_prompt = '''
You are an expert in SQL. I will give you a natural language question and a database schema, 
please help me generate the corresponding SQL query with no further explaination.
'''
three_shots_SQL_generation_prompt = '''
Here is some examples of EASY, MEDIUM and HARD SQL queries.
SELECT count(*) FROM singer 
SELECT avg(weight) ,  pettype FROM pets GROUP BY pettype
SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')
'''
checker_prompt = '''
Please help me generate the corresponding SQL query with no further explaination.
'''


#################### 1. Set up  ####################
#----------------------------------------------------------------------------------------------------------

API_KEY = "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" # 自己的
# API_KEY = "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"  # 买的
# API_KEY = "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" # gpt4 孙哥
os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

#changed
task = 'CoSQL' # 1 for CoSQL, 2 for Spider
if task == 'CoSQL':
    path_to_CoSQL = "/Users/yan/Desktop/text2sql/cosql_dataset"
    DATASET_SCHEMA = path_to_CoSQL+"/tables.json"
    DATASET = path_to_CoSQL+"/sql_state_tracking/cosql_dev.json"
    OUTPUT_FILE_1 = "./predicted_sql.txt"
    OUTPUT_FILE_2 = "./gold_sql.txt"
    DATABASE_PATH = path_to_CoSQL+"/database"
else:
    path_to_Spider = "/Users/yan/Desktop/text2sql/spider"
    DATASET_SCHEMA = path_to_Spider + "/tables.json"
    DATASET = path_to_Spider + "/dev.json"
    OUTPUT_FILE_1 = "./Spider/predicted_sql.txt"
    OUTPUT_FILE_2 = "./Spider/gold_sql.txt"
    DATABASE_PATH = path_to_Spider + "/database"


# set max tokens limit
MAX_TOKENS = 4096
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
encoding = tiktoken.encoding_for_model(model_name)
# count the token
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# load dataset
def load_data(DATASET):
    return pd.read_json(DATASET)


def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
  return output
def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output
def find_primary_keys_MYSQL_like(db_name):
  df = spider_primary[spider_primary['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output
def creatiing_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def SQL_checker(sql, database):
    # sql be like: "SELECT * FROM car_1 WHERE car_1.id = 1"
    # database is the path to local xxx.sqlite
    # the function of this part is to check if the sql is valid, if not, return the error message
    path = DATABASE_PATH + '/' + database + '/' + database + '.sqlite'
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(path)
        # Create a cursor object to execute the SQL query
        cursor = conn.cursor()
        # Execute the SQL query
        cursor.execute(sql)
        # Commit the transaction and close the connection
        conn.commit()
        conn.close()
        # Return a success message if the SQL query is valid
        prompt =  "The SQL query is valid in grammar."
        checker = False
    except sqlite3.Error as e:
        # Return the error message if the SQL query is not valid
        instruction = f"""#### the sql generated by you: {sql}, has error like :{e} , please fix the error and generate again. \n"""
        fields = find_fields_MYSQL_like(database)
        fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
        fields += "Primary_keys = " + find_primary_keys_MYSQL_like(database)
        prompt = instruction + fields + checker_prompt
        checker = True
    return prompt, checker

def GPT4_generation(prompt):
    limit_marker = False
    try:
        response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        n = 1,
        stream = False,
        temperature=0.0,
        max_tokens=600,
        top_p = 1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        )
        return response['choices'][0]['message']['content'], limit_marker
    except openai.error.RateLimitError as e:
        print(f"RateLimitError: {e}")
        limit_marker = True
        fake_SQL = "SELECT COUNT(*) FROM singer"
        return fake_SQL,limit_marker


def save_breaker(breaker):
    with open("breaker.txt", "w") as f:
        f.write(str(breaker))

# Function to load the breaker value from a file
def load_breaker():
    if os.path.exists("breaker.txt"):
        with open("breaker.txt", "r") as f:
            breaker =  int(f.read())
            if breaker > 1037:
                breaker = 0
            else:
                breaker = breaker
            return breaker
    return 0



if __name__ == '__main__':
###########################################################################################
    spider_schema,spider_primary,spider_foreign = creatiing_schema(DATASET_SCHEMA)
    val_df = load_data(DATASET)
    CODEX = []
    # test_SQL = "SELECT COUNT(*) FROM singer"
    # test_db = 'concert_singer'
    # print(SQL_checker(test_SQL, test_db))
    breaker = load_breaker() # mark the breaker point of chatgpt
    print("breaker is: ", breaker)
    for index, row in val_df[breaker:].iterrows():
        #if index < 405: continue #for testing
        print(f"index is {index}")
        print(row['query'])
        print(row['question'])
        question = row['question']
        db_id = row['db_id']
        sql = row['query']
        schema = find_fields_MYSQL_like(db_id)+'\n'+"foreign key:"+find_foreign_keys_MYSQL_like(db_id)+'\n'+"primary key:"+find_primary_keys_MYSQL_like(db_id)
        # print(schema)
        message = SQL_generation_prompt + "Question:"+question + "\ndatabase:"+ db_id + "\ndatabase chema:"+schema+ three_shots_SQL_generation_prompt
        # print(message)
        SQL,limit_marker = GPT4_generation(message)
        if limit_marker:
            print("break at index: ", breaker)
            break
        else:
            result_message,checker = SQL_checker(SQL, db_id)
            if checker:
                print(result_message)
                SQL,_ = GPT4_generation(result_message)
            else:
                print(result_message)
            SQL = SQL.replace('\n', ' ')
            breaker += 1
            CODEX.append([row['question'], SQL, row['query'], row['db_id']])
            # break
    df = pd.DataFrame(CODEX, columns=['NLQ', 'PREDICTED SQL', 'GOLD SQL', 'DATABASE'])
    results = df['PREDICTED SQL'].tolist()
    with open(OUTPUT_FILE_1, 'a') as f:
        for line in results:
            f.write(f"{line}\n")

    task = 'CoSQL'
    if task == 'CoSQL':
        dataset = pd.read_json(DATASET)
        gold = []
        for index, row in dataset[:index].iterrows():
            dict_round = {}
            dict_round['query'] = row['interaction'][0]['query']
            dict_round['db_id'] = row['database_id']
            gold.append(dict_round)
    else:
        dataset = pd.read_json(DATASET)
        gold = []
        for index, row in dataset[:index].iterrows():
            dict_round = {}
            dict_round['query'] = row['query']
            dict_round['db_id'] = row['db_id']
            gold.append(dict_round)

    with open(OUTPUT_FILE_2, 'a') as f:
        for item in gold:
            f.write(f"{item['query']}\t{item['db_id']}\n")
    save_breaker(breaker)