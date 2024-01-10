# GPT-3 SQL integration with python and pandas for tabular data integration and understanding capability  
import sqlite3
import pandas as pd
import numpy as np
import openai

# Connect to the GPT-3 API
openai.api_key = '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'

######################################################
# Connector to the database
conn = sqlite3.connect('chatgpt.db')


# Create a function to get the table columns
def get_table_columns(table_name):
  cursor.execute("PRAGMA table_info({})".format(table_name))
  columns = cursor.fetchall()
  print(columns)
  return [column[1] for column in columns]

# Generate the SQL query
def generate_sql_query(table_name,text,columns,company_name):
  prompt = """ you are a ChatGPT Language Model that can generate SQL Queries. \
                please provice a natural language input text and i will generate the corresponding \
                SQL query for you.Keep in mind that the company_name {}  may be provided in different variations,\
                such as lowercase, uppercase, or with additional characters. Example: Consider 'astral' as 'Astral'.Your query should be able to handle\
                these variations accurately. Table Name is {} and corresponding columns are {}. \
                Input:{}\nSQL Query:
                
                """.format(company_name,table_name,columns,text)\
                  
                
                    
  print(prompt)
  request = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-0301",
  messages =[{"role":"system","content":"You are a ChatGPT Language Model that can generate SQL Queries"},
            {"role":"user","content":prompt}]

  )
  sql_query = request['choices'][0]['message']['content']
  return sql_query

# Execute the SQL query
def execute_sql_query(query):
  cursor.execute(query)
  result = cursor.fetchall()
  conn.commit()
  return result



# Read the data and Connect to the database
df = pd.read_excel(r"C:\Users\ashutosh.somvanshi\PVC_Trend_Analysis\Gpt_Data_understanding Capability\gpt_sql\Company data.xlsx")
conn = sqlite3.connect('chatgpt.db')
cursor = conn.cursor()
df.to_sql('site_data',conn,if_exists='replace',index=False)
# conn.close()


# Create the table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS site_data(
        Name TEXT,
        Net_Profit_Quarter FLOAT,
        Sales_Quarter FLOAT,
        Profit_Growth_3_Years FLOAT,
        Profit_Growth_5_Years FLOAT,
        Sales_Growth_5_Years FLOAT,
        Sales_Growth_3_Years FLOAT
    )
""")

def main():
    cursor = conn.cursor()
    # Ask the user for the input text
    text = str(input("Ask the question to get information about the company you want to know "))
    table_name = 'site_data'
    columns = get_table_columns(table_name)
    sql_query = generate_sql_query(table_name,text,columns,list(df.Name))
    print("Generated SQL Query: ",sql_query)
    if sql_query:
        result = execute_sql_query(sql_query)
        print("ChatGPT Response => ",result)
     
    cursor.close()
    conn.close()
    
    
if __name__ == '__main__': 
    main()  
    
    