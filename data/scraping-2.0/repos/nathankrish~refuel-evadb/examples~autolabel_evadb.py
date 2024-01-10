import os
import evadb
from my_secrets import OPENAI_KEY
# provide your own OpenAI API key here
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

cursor = evadb.connect().cursor()
print("Connected to EvaDB")

from autolabel import get_data
get_data('walmart_amazon')

# Create function in EvaDB

create_function_query = f"""CREATE FUNCTION IF NOT EXISTS RefuelAutolabel
            IMPL  './functions/refuel_autolabel.py';
            """
cursor.query("DROP FUNCTION IF EXISTS RefuelAutolabel;").execute()
cursor.query(create_function_query).execute()
print("Created Function")

# Plan Autolabeling Task

query= f""" SELECT RefuelAutolabel("plan", 'config_banking.json', 'seed.csv');"""
result = cursor.query(query).execute()

# Run Autolabeling Task
"""
use the function arguments to specify
    1. the mode to execute in (plan, run, explain)
    2. config file
    3. dataset file
    4. (optional) output file
    5. (optional) max items
    6. (optional) start index
    7. (optional) skip eval
"""

query_with_output = f""" SELECT RefuelAutolabel("run", 'config_banking.json', 'seed.csv', 'output.csv');"""

query= f""" SELECT RefuelAutolabel("run", 'config_banking.json', 'seed.csv');"""
result = cursor.query(query).execute()

query= f""" SELECT RefuelAutolabel("run", 'config_walmart.json', 'seed.csv', 'output.csv', '10', '1', 'false');"""
result = cursor.query(query).execute()

# Explain Autolabeling Task
query= f"""SELECT RefuelAutolabel("explain", 'config_banking.json', 'seed.csv');"""
result = cursor.query(query).execute()

# Bring labeled dataset from csv into EvaDB
drop_query = "DROP TABLE IF EXISTS MyCSV"
cursor.query(drop_query).execute()

query1 = """CREATE TABLE IF NOT EXISTS MyCSV (
                example TEXT(1000),
                label TEXT(100),
                explanation TEXT(1000)
            );"""

query2 = "LOAD CSV 'test_banking.csv' INTO MyCSV;"
query3 = "SELECT * FROM MyCSV;"

result = cursor.query(query1).execute()
result = cursor.query(query2).execute()
result = cursor.query(query3).execute()

print(result)