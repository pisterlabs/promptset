import openai
import pandas as pd
import json
from dotenv import load_dotenv
import os
import pyodbc

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY3')

# Connect to the database
server = 'transimdb-prod.be2231f9be26.database.windows.net' # Replace with your SQL Server name
database = 'MOVE_VP_New' # Replace with your database name
username = os.getenv('MSSQL_USERNAME') # Replace with your username
password = os.getenv('MSSQL_USERNAME') # Replace with your password
driver= '{ODBC Driver 17 for SQL Server}' # Replace with your driver name

# Create the connection string
conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"

# Connect to the database
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Execute the SQL query
query = """
select top 1 * From v_test
"""

cursor.execute(query)
data = cursor.fetchall()
# Convert the data to a list of dictionaries
columns = [column[0] for column in cursor.description]
json_data = []
for row in data:
    json_data.append(dict(zip(columns, row)))

# Convert the list of dictionaries to JSON format
json_str = json.dumps(json_data)

# Print or save the JSON data
# print(json_str)
# headers = ['PartNumber', 'ManufacturerName', 'PartDescription', 'FeatName', 'FeatureValue']
# print(headers)

# with open('table_copy.json', 'r') as file:
#     data = json.load(file)

def apply_model(query):
    prompt = """Please regard the following table: {}

    The table name is v_test. Use ' as the quote character. Quote column aliases with ". Write a SQL query to answer the following question: {}""".format(json_str, query)
    print(prompt)

    request = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=3500
    )
    sql_query = request.choices[0].text
    print("===> {}: {}\n".format(query, sql_query))    

    cursor = conn.cursor()
    cursor.execute(sql_query)
    result = cursor.fetchone()
    conn.close()

    return result

print(apply_model("What's the lowest feature value?"))