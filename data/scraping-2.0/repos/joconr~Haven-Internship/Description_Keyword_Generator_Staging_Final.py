import importlib
import subprocess

# List of required packages
packages = ["os", "openai", "psycopg2", "bs4", "getpass", "datetime"]

# Loop through each package and check if it can be imported
for package in packages:
    try:
        importlib.import_module(package)
        print(f"{package} package is already installed")
    except ImportError:
        # If the package can't be imported, use pip to install it
        print(f"{package} package is not installed, installing now...")
        subprocess.check_call(["pip", "install", package])

import psycopg2
from bs4 import BeautifulSoup
import os
import openai
from getpass import getpass
from datetime import datetime

# Connect to the Postgres database
conn = psycopg2.connect(
    host="haven-staging-db-do-user-10050983-0.b.db.ondigitalocean.com",
    port=25060,
    dbname="staging",
    user="chatgpt_user", # Change user to your Postgres user
    password="--" # Change password to your Postgres password
)
cursor = conn.cursor()

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY_HERE"

# Retrieve the description column data, SQL can be customized here
# I've set this to only return 10 specific opportunities for testing purposes, including one opportunity with no description
cursor.execute("SELECT id, description FROM opportunity o WHERE o.ID IN (73810, 77956, 24642, 46126, 62654, 9930, 46600, 55576, 20691, 58681);")
descriptions = cursor.fetchall()

# Loop through each description, extract the text using BeautifulSoup, and generate a summary using OpenAI API
for opportunity_id, description in descriptions:
    
    # If the description is empty, set a default summary and token
    if not description or description.strip() == "":
        print(f"Empty description for opportunity_id: {opportunity_id}")
        default_summary = "No description provided." # The default summary for no description can be customized here
        cursor.execute("""
            UPDATE opportunity SET gpt_description = %s, gpt_description_token = 0 WHERE id = %s
        """, (default_summary, opportunity_id))
        conn.commit()
        continue
    
    # If the description is not empty, generate a summary
    soup = BeautifulSoup(description, 'html.parser')
    text = soup.get_text()
    truncated_text = text[:15000] # The max number of characters for the API call can be customized here, but be careful as the token limit is 4096 (16384 characters for both input and output)

    # Generate a summary using OpenAI API
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You summarize opportunities from various government industries."}, # Adjusting system prompt can be customized here, but I've found it isn't as effective as the user prompt
            {"role": "user", "content": f'Summarize the contract work detail in the following text.: "{truncated_text}"'} # Adjusting user prompt can be customized here
        ],
        temperature=0,
    )

    # Print the summary and the number of tokens used
    summary = response['choices'][0]['message']['content'] 
    tokens = response['usage']['total_tokens']  # The number of tokens used is the cost of the API call
    print(summary)
    print(tokens)

    try:
            # Update the opportunity table with the summary and token count
            cursor.execute("""
                UPDATE opportunity SET gpt_description = %s, gpt_description_token = %s WHERE id = %s
            """, (summary, tokens, opportunity_id))
            conn.commit()
            print(f"Summary inserted for opportunity_id: {opportunity_id}")
    # If there is an error inserting into the table, print the error message
    except Exception as e:
        print(f"Error updating opportunity table for opportunity_id {opportunity_id}: {e}")
    
conn.close() # Comment this out if you want to test the keyword generator script below

# !! THE FOLLOWING CODE IS FOR THE KEYWORD GENERATOR SCRIPT, COMMENT OUT THE ABOVE CODE AND UNCOMMENT THE BELOW CODE TO TEST THE KEYWORD GENERATOR SCRIPT !!

# # Retrieve the data from the 'gpt_description' table
# cursor.execute("SELECT id, gpt_description FROM opportunity o WHERE o.ID IN (73810, 77956, 24642, 46126, 62654, 9930, 46600, 55576, 20691, 58681)")
# descriptions = cursor.fetchall()

# # Loop through each description and generate keywords using OpenAI gpt-3.5-turbo model
# for desc in descriptions:
#     cleanDesc = desc[1]

#     # Check if the description is set to "No description provided."
#     if cleanDesc == "No description provided.":
#         print(f"No keywords generated for opportunity_id: {desc[0]} (no description provided)")
#         continue

#     response = openai.ChatCompletion.create(
#     model='gpt-3.5-turbo',
#     messages=[
#         {"role": "system", "content": "You are an assistant generating keywords for government contracting descriptions."},
#         {"role": "user", "content": f'Generate keywords for the following prompt. Keywords should relate ONLY to the work detail of the project involved. Ignore any names, dates, and emails. Separate keywords by commas: "{cleanDesc}"'}
#     ],
#     temperature=0,
#     )
#     keywords = response['choices'][0]['message']['content']
#     tokens = response['usage']['total_tokens']
#     keywords_list = [keyword.strip() for keyword in keywords.split(',')]  # Delimit and split the keywords by commas
#     print(keywords_list)
#     print(tokens)

#     # Update the opportunity table with the gpt_keyword_token attribute
#     cursor.execute("""
#         UPDATE opportunity SET gpt_keyword_token = %s WHERE id = %s
#     """, (tokens, desc[0]))
    
#     # Insert the generated keyword data into the 'keyword' and 'opportunity_keyword' tables
#     for keyword in keywords_list:
#         # Check if the keyword already exists in the 'keyword' table
#         cursor.execute("SELECT id FROM keyword WHERE type = 'opportunity' AND name = %s", (keyword,))
#         keyword_id = cursor.fetchone()

#         # If the keyword doesn't exist, insert it into the 'keyword' table and get its ID
#         if keyword_id is None:
#             cursor.execute(
#                 "INSERT INTO keyword (type, name) VALUES ('opportunity', %s) RETURNING id",
#                 (keyword,)
#             )
#             keyword_id = cursor.fetchone()[0]

#         # Check if there's an existing entry in the 'opportunity_keyword' table
#         cursor.execute(
#             "SELECT * FROM opportunity_keyword WHERE opportunity_id = %s AND keyword_id = %s",
#             (desc[0], keyword_id)
#         )
#         existing_entry = cursor.fetchone()

#         # If there's no existing entry, insert the association into the 'opportunity_keyword' table
#         if existing_entry is None:
#             cursor.execute(
#                 "INSERT INTO opportunity_keyword (opportunity_id, keyword_id) VALUES (%s, %s)",
#                 (desc[0], keyword_id)
#             )

#     # commit changes to the database
#     conn.commit()

# conn.close()
