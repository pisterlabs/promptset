import sqlite3
import openai

# Connect to SQLite database
conn = sqlite3.connect('db.sqlite3')
cursor = conn.cursor()

# SQL Query
# cursor.execute('''SELECT u.date_of_joining
# FROM api_inventories i
# INNER JOIN api_user u ON i.alloted_to_id = u.user_id 
# WHERE i.alloted_to_id = (SELECT i2.alloted_to_id 
#                          FROM api_inventories i2 
#                          GROUP BY i2.alloted_to_id 
#                          ORDER BY COUNT(i2.alloted_to_id) DESC 
#                          LIMIT 1)''')
# rows =cursor.fetchall()
# for row in rows:
#     print(row)

# exit(0)

openai.api_key = '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'


# Function to get table columns from SQLite database
def get_table_columns(table_name):
    cursor.execute("PRAGMA table_info({})".format(table_name))
    columns = cursor.fetchall()
    return [column[1] for column in columns]



# Function to generate SQL query from input text using ChatGPT
def generate_sql_query(table_name, table_name_2, table_name_3, text, columns, columns_2, columns_3):
    prompt = """You are a ChatGPT language model that can generate SQL queries. Please provide a natural language input text, and I will generate the corresponding SQL query for you.The table name are {}, {} and {} corresponding columns are {}, {} and {}.\nInput: {}\nSQL Query:""".format(
        table_name, table_name_2, table_name_3, columns, columns_2, columns_3, text)
    print(prompt)
    request = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    sql_query = request['choices'][0]['message']['content']
    print(sql_query)
    return sql_query

# Function to execute SQL query on SQLite database


def execute_sql_query(query):
    cursor.execute(query)
    result = cursor.fetchall()
    return result


text = "Select * from inventory?"

file_path = "./question.txt"

table_name = 'api_inventories'
table_name_2 = "api_user"
table_name_3 = 'api_category'
columns = get_table_columns(table_name)
columns_2 = get_table_columns(table_name_2)
columns_3 = get_table_columns(table_name_3)
sql_query = generate_sql_query(table_name, table_name_2, table_name_3, text, columns, columns_2, columns_3)
if sql_query:
    result = execute_sql_query(sql_query)
    print("ChatGPT Response=>", result)

with open(file_path, "a") as file:
    file.write(text + '====>\n' + sql_query + '\n\n\n')


# Close database connection
file.close()
cursor.close()
conn.close()