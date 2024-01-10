from dotenv import load_dotenv
import os
import openai
import subprocess
import sqlite3
import csv

load_dotenv()

with open('database-info.txt', 'r') as file:
    database_info = file.read()

openai.api_key = os.getenv("OPENAI_API_KEY")

# e.g Give me a list of all customer names with the count of unique invoices and the sum of unit price where the sum of unit price is greater than 45. Order by customer name
user_query = input("Please enter your query: ")

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are a sql expert in data analysis"},
        {"role": "user", "content": 
        f'''Pretend you are a sql data analsyis,

            My database break down is as follows:

            {database_info}
            I want you to give only the sql code as a output so for example:

            Input: How do i select all the customers first names
            Output: 
            SELECT FirstName FROM customers;

            input: Update a albums name to "new name" where its id is 101
            Output:
            UPDATE albums
            SET Title = 'new name'
            WHERE Albumid = 101;

            -----


            {user_query}
            
'''}
        ]
    )

query = chat_completion['choices'][0]['message']['content']

print(query)

conn = sqlite3.connect('chinook.db')
cursor = conn.cursor()

cursor.execute(query)

rows = cursor.fetchall()
print(rows)

with open('extracted-data.csv', 'w', newline='') as f:
    #f.write("\n")
    writer = csv.writer(f)
    writer.writerows(rows)

conn.close()


chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are a expert in explaining data analysis to non technical audience"},
        {"role": "user", "content": 
        f''' Given this user query:

            {user_query}

            And this output:
            {rows}

            Explain the output data to me, dont go into much technical details, your tagret audience is non technical. Just explain the output data and context

'''}
        ]
    )

query_explain = chat_completion['choices'][0]['message']['content']

print(query_explain)
