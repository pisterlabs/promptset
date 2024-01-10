import evadb
import psycopg2
import pandas as pd
from tabulate import tabulate  # Ensure you have the tabulate library installed
import datetime
import stock_analysis
import openai

# Database connection parameters
db_params = {
    'user': 'postgres',
    'password': 'bbqporkbun25$',
    'host': 'localhost',
    'database': 'postgres'
}

# Connect to the PostgreSQL database
# connection = psycopg2.connect(**db_params)
# cursor = connection.cursor()

cursor = evadb.connect().cursor()

# SQL command to fetch all data from the stock_data table
select_data_sql = "USE postgres_stock_data {SELECT * FROM stock_data WHERE stock_symbol = 'GOOGL' LIMIT 10}"

# Execute the SQL command
# print(cursor.query(select_data_sql).df())
# print(cursor.query("SHOW FUNCTIONS;").df())

# OPENAI STUFF
print("OpenAI stuff")

client = openai.OpenAI(
    api_key='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',
)

messages = [ {"role": "system", "content":  
              "You are an intelligent assistant."} ]

# while True: 
#     message = input("User : ")
#     if message == "exit":
#         break
#     elif message: 
#         messages.append( 
#             {"role": "user", "content": message}, 
#         ) 
#         chat = client.chat.completions.create( 
#             model="gpt-3.5-turbo", messages=messages 
#         ) 
#     reply = chat.choices[0].message.content 
#     print(f"ChatGPT: {reply}") 
#     messages.append({"role": "assistant", "content": reply})


qTSLA = "USE postgres_stock_data {SELECT * FROM stock_data WHERE stock_symbol = 'TSLA' LIMIT 10}"

tslaLast10 = cursor.query(qTSLA).df()

# print(tslaLast10)

exit()

print("AI STUFF")
# df = cursor.query("SELECT StockPriceForecast(30) ORDER BY DATE DESC LIMIT 50;").df()

query = f"""
USE postgres_stock_data {{
    SELECT stock_symbol, date, close
    FROM stock_data
    WHERE stock_symbol = 'NVDA'
    ORDER BY date DESC
    LIMIT 50
}}
"""
# df = cursor.query(query).df()
# print(df)

# stock_analysis.forecast_price("NVDA", 50)

df = cursor.query("SELECT ForecastNVDA(30) ORDER BY DATE DESC;").df()
print(df)

# stock_name = query.iloc[0, 0]
# stock_date = query.iloc[0, 1].strftime('%Y-%m-%d')
# stock_price = query.iloc[0, 2]
# formatted_date = datetime.datetime.strptime(stock_date, '%Y-%m-%d').date().strftime('%A, %B %d')
# formatted_price = "{:.2f}".format(stock_price)

# output_sentence = f"The projected stock of 1 share of {stock_name} is ${formatted_price} on {formatted_date}."
# print(output_sentence)


# cursor.execute(select_data_sql)


# Close the cursor and connection
cursor.close()
# connection.close()
