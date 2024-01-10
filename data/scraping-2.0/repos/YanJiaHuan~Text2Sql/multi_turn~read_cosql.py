import pandas as pd
import time
import openai
import os
import sys
import tiktoken
import sqlite3
from Bard import Chatbot


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

zero_shots_SQL_generation_prompt = '''
Sorry, I won't give you any examples. Please generate based on your own semantic parsing ability.
'''

one_shot_Cosql_prompt_without_explain = '''
Here is a sample of multi-turn text2sql for you to understand the task.
Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]

foreign key:[course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
primary key:[classroom.building,department.dept_name,course.course_id,instructor.ID,section.course_id,teaches.ID,student.ID,takes.ID,advisor.s_ID,time_slot.time_slot_id,prereq.course_id]

Iteration 1:
Question: Find out the average salary of professors?
SELECT avg ( salary )  FROM instructor

Iteration 2: # iteration 2 will see the question and sql in iteration 1
Question: Find the average salary of the professors of each department?
SELECT avg ( salary ) , dept_name FROM instructor GROUP BY dept_name

Iteration 3: # iteration 3 will see the questiones and sqls in iteration 2 and 1
Question: Which department has the highest average salary of professors?
SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg ( salary )  DESC LIMIT 1

Quesion: show the train name and station name for each train.


'''


checker_prompt = '''
Please help me generate the corresponding SQL query with no further explaination.
'''

Contextual_prompt = '''
Now I will give you some context (question and your own answer). Please generate the corresponding SQL query with no further explaination.
'''

#################### 1. Set up  ####################
#----------------------------------------------------------------------------------------------------------

# API_KEY = "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" # 自己的
API_KEY = "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"  # 买的
# API_KEY = "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" # gpt4 孙哥
os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

#changed
task = 'CoSQL' # 1 for CoSQL, 2 for Spider
if task == 'CoSQL':
    path_to_CoSQL = "./cosql_dataset"
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

import time

def GPT4_generation(prompt):
    limit_marker = False
    fake_SQL = "SELECT COUNT(*) FROM singer"

    while True:
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
            print("Sleeping for 20 seconds...")
            time.sleep(20)
            print("Retrying...")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return fake_SQL, limit_marker

token = "Wgj-oa5yHxfmjo0lLybtWGLiWYoKTZ07NXcUiaPiUHmtQQiAKlfzNTOA9lwqmCz2N0qGFg."
chatbot = Chatbot(token)
def Bard_generation(prompt):
    a = chatbot.ask(prompt)
    answer = a['content']
    print(answer)
    return answer

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
    # load the data
    spider_schema,spider_primary,spider_foreign = creatiing_schema(DATASET_SCHEMA)
    val_df = load_data(DATASET)
    Log_content = []
    for index,sample in val_df.iterrows():
        print('index:',index)
        db_id = sample['database_id'] # e.g.'car_1'
        question_final = sample['final']['utterance'] # e.g.'How many car models are produced by each maker? List the count and the maker full name.'
        query_final = sample['final']['query'] # e.g.'SELECT COUNT(*) FROM car_1 WHERE car_1.id = 1'
        schema = find_fields_MYSQL_like(db_id) + '\n' + "foreign key:" + find_foreign_keys_MYSQL_like(
            db_id) + '\n' + "primary key:" + find_primary_keys_MYSQL_like(db_id)  #
        '''  
        schema: Table car_makers, columns = [*,Id,Maker,FullName,Country]
        Table car_names, columns = [*,MakeId,Model,Make]
        Table cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]
        Table continents, columns = [*,ContId,Continent]
        Table countries, columns = [*,CountryId,CountryName,Continent]
        Table model_list, columns = [*,ModelId,Maker,Model]

        foreign key:[countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]
        primary key:[continents.ContId,countries.CountryId,car_makers.Id,model_list.ModelId,car_names.MakeId,cars_data.Id]
        '''
        # for first round:
        # input: question+db_id+schema+three_sqls
        # output: sql
        # for other rounds and final round:
        # input: question + message + generated_sql
        # output: sql
        message = ''
        old_message = ''
        history = {}
        tmp = {}
        SQLs_temp_pred = []
        SQLs_temp_gold = []
        tmp['question'] = question_final
        for round, dialog in enumerate(sample['interaction']): # assueme the goal it to output the final sql by using final question and dialog information
            print(f'The {round} round of dialog in sample {index}:') # each sample has at least 1 previous conversation
            question_round = dialog['utterance']
            query_round = dialog['query']
            if round == 0:
                old_message = message + \
                          SQL_generation_prompt + \
                          "\ndatabase:" + db_id + \
                          "\ndatabase chema:" + schema + \
                          "\nSome samples to text2sql:" + one_shot_Cosql_prompt_without_explain
                message = message + \
                          SQL_generation_prompt + \
                          "\ndatabase:" + db_id + \
                          "\ndatabase chema:" + schema + \
                          "\nSome samples to text2sql:" + one_shot_Cosql_prompt_without_explain+ \
                          "\nQuestion:" + question_round + \
                          "\nOutput:"
            else:
                message = old_message + \
                          Contextual_prompt + \
                          "\nThis is previous question:" + history['question'] + \
                          "\nThis is your previous generated SQl:" + history['query']+ \
                          "\nQuestion:" + question_round + \
                          "\nOutput:"
                old_message = old_message + \
                            "\nThis is previous question:" + history['question'] + \
                            "\nThis is your previous generated SQl:" + history['query']
            print('message:',message)
            SQL= Bard_generation(message)
            SQL = SQL.replace('\n',' ')
            print('\nGPT generated SQL:',SQL+'\n')
            history['question'] = question_round
            history['query'] = SQL
            '''
            save the log and generated sql, gold sql in some file: may need to use some process as the response is like: 
            SELECT car_names.Model, COUNT(cars_data.Id) AS popularity
            FROM car_names
            JOIN cars_data ON cars_data.Id = car_names.MakeId
            GROUP BY car_names.Model
            ORDER BY popularity DESC;
            There are '\n' in line, and I don't want it
            '''
            SQLs_temp_pred.append(SQL)
            SQLs_temp_gold.append(query_round+'\t'+db_id)
        # this loop will focus on the final round, which is the 'final' in dataset
        with open ('./predicted_sql.txt','a') as f:
            for line in SQLs_temp_pred:
                f.write(line+'\n')
        with open ('./gold_sql.txt','a') as f:
            for line in SQLs_temp_gold:
                f.write(line+'\n')

# CUDA_VISIBLE_DEVICES=7 python read_cosql.py