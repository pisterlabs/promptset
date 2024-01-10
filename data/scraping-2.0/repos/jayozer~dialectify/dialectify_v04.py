import re
import os
import streamlit as st
import openai
from dotenv import load_dotenv
# loads .env file located in the current directory
load_dotenv() 

import sqlparse
import random
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML, Name

import time

#openai.api_key = os.getenv('OPENAI_API_KEY')

# Declare word_map and masked_converted_sql as global variables
word_map = {}
masked_converted_sql = ""


# Set the app title
st.title("Dialectify SQL")

# Create the input boxes for the SQL code and the SQL dialects
openai.api_key = st.text_input("Enter API Key:")
sql = st.text_area("Enter SQL Code")
from_sql = st.selectbox("From SQL:", ["Transact-SQL", "MySQL", "PL/SQL", "PL/pgSQL", "SQLite", "Snowflake"])
to_sql = st.selectbox("To SQL:", ["Transact-SQL", "MySQL", "PL/SQL", "PL/pgSQL", "SQLite", "Snowflake"])

# Extract fields for masking - identifier_set

def get_identifiers(sql):
    parsed_tokens = sqlparse.parse(sql)[0]
    identifier_set = set()

    reserved_words = ['TOP', 'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC']

    def process_identifier(token, identifier_set):
        # Get the real name of the token
        identifier_name = token.get_real_name()
        # If the real name contains a dot, split it on the dot and take the second part
        if '.' in identifier_name:
            identifier_name = identifier_name.split('.')[1]
        # Add the identifier name to the set
        identifier_set.add(identifier_name)
    def process_function_arguments(token, identifier_set):
        if isinstance(token, sqlparse.sql.Identifier):
            process_identifier(token, identifier_set)
        elif isinstance(token, sqlparse.sql.IdentifierList):
            for identifier in token.get_identifiers():
                if isinstance(identifier, sqlparse.sql.Identifier):
                    process_identifier(identifier, identifier_set)
        elif isinstance(token, sqlparse.sql.Parenthesis):
            for subtoken in token.tokens:
                process_function_arguments(subtoken, identifier_set)

    def add_identifiers_from_function(token, identifier_set):
        for subtoken in token.tokens:
            process_function_arguments(subtoken, identifier_set)

    def process_where(token, identifier_set):
        for subtoken in token.tokens:
            if isinstance(subtoken, sqlparse.sql.Comparison):
                process_identifier(subtoken.left, identifier_set)
            elif isinstance(subtoken, sqlparse.sql.Identifier):
                process_identifier(subtoken, identifier_set)

    for token in parsed_tokens.tokens:
        if isinstance(token, sqlparse.sql.Function):
            add_identifiers_from_function(token, identifier_set)
            continue
        if token.value.upper() in reserved_words:
            continue
        if isinstance(token, sqlparse.sql.IdentifierList):
            for identifier in token.get_identifiers():
                if isinstance(identifier, sqlparse.sql.Identifier):
                    process_identifier(identifier, identifier_set)
        elif isinstance(token, sqlparse.sql.Comparison):
            process_identifier(token.left, identifier_set)
        elif isinstance(token, sqlparse.sql.Where):
            process_where(token, identifier_set)
        elif isinstance(token, sqlparse.sql.Identifier):
            process_identifier(token, identifier_set)
    return list(identifier_set)

def sql_masking(identifiers, sql):
    """
    This function takes in a list of identifiers and an SQL query as input, and replaces the identifiers in the SQL query with random words.
    """
    # Create a dictionary to store the mapping between original identifiers and masked words
    word_map = {}
    
    # Loop through each identifier in the list of identifiers
    for identifier in identifiers:
        # Generate a random word to replace the original identifier
        random_word = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=len(identifier)))
        
        # Add the mapping to the dictionary
        word_map[identifier] = random_word
        
        # Replace the original identifier with the random word in the SQL string
        sql = re.sub(r'\b{}\b'.format(identifier), random_word, sql)
    
    # Return the masked SQL string and the word map
    return sql, word_map


# Open Ai piece
# Create a sidebar in Streamlit
st.sidebar.title("Dialectify SQL")

# Add a selectbox for max_tokens to the sidebar with a text box
# User can enter the integer value of max_tokens and select it from the dropdown.
# Also add a note to the sidebar explaining the purpose of max_tokens.
max_tokens = st.sidebar.selectbox("Enter Max Tokens", [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192], index=0) 
st.sidebar.markdown("GPT-4 has a maximum token limit of 8,192 tokens (equivalent to ~6000 words), whereas GPT-3.5's 4,000 tokens (equivalent to 3,125 words)")

# model_choice = st.sidebar.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4"])
model_choice = st.sidebar.radio("Model:", ["gpt-3.5-turbo", "gpt-4"])
temperature = st.sidebar.selectbox("Temperature:", [0.1, 0.2, 0.3, 0.9], index=1)

def sql_dialectify (from_sql, to_sql, masked_sql, max_tokens=max_tokens, model_choice=model_choice, temperature = temperature):
    completion = openai.ChatCompletion.create(
        model=model_choice,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": 'Act as CODEX ("COding DEsign eXpert"), an AI expert in SQL programming languages, focusing on accurate SQL dialect conversions.'},
            {"role": "system", "content": 'You are proficient in various SQL dialects, including Transact-SQL, MySQL, PL/SQL, PL/pgSQL, SQLite, and Snowflake SQL. Your expertise lies in converting scripts between these dialects while maintaining the original functionality and integrity.'},
            {"role": "system", "content": 'Your task is to convert a given SQL script from one SQL dialect to another, ensuring that the converted script retains the same functionality and addresses any incompatibilities or differences between the dialects.'},
            {"role": "system", "content": f'The source SQL dialect is "{from_sql}", and the target SQL dialect is "{to_sql}". Perform an accurate conversion that addresses data types, functions, and syntax differences between the two dialects.'},
            {"role": "system", "content": 'Adhere to coding best practices by writing clean, modular code that includes proper security measures and follows established design patterns.'},
            {"role": "system", "content": f'Identify and address differences in data types and functions between the {from_sql} and {to_sql} dialects. For data types or functions without a direct equivalent, select the most suitable alternative while maintaining the original functionality.'},
            {"role": "system", "content": 'Present your solution in two sections. In the first section, provide the converted SQL query in a code block, titled "### Converted SQL:". In the second section, include any comments and explanations of the changes as bullet points, titled "### Conversion details:".'},
            {"role": "user", "content": f'Accurately convert the following SQL code from "{from_sql}" to "{to_sql}", maintaining the original functionality and addressing any incompatibilities or differences: "\n\n{masked_sql}"'},
]
    )
    converted_sql = completion.choices[0].message.content
    return converted_sql



# Demask Converted SQL
def demasking(word_map, masked_sql):
    """
    This function takes in a word map and a masked SQL string as input and replaces the masked words with their original words.
    """
    demasked_sql = masked_sql

    # Loop through each key-value pair in the word map
    for original_word, masked_word in word_map.items():
        # Replace the masked word with the original word in the SQL string
        demasked_sql = re.sub(r'\b{}\b'.format(masked_word), original_word, demasked_sql)
    
    # Return the demasked SQL string
    return demasked_sql

# Extract tables to append _view

def extract_table_identifiers(token_stream):
    for item in token_stream:
        # Check if the token stream contains an identifier.
        if isinstance(item, Identifier):
            # Return the real name of the identifier.
            yield item.get_real_name()
        # Check if the token stream contains an identifier list.
        elif isinstance(item, IdentifierList):
            # Loop over the identifiers in the identifier list.
            for identifier in item.get_identifiers():
                # Check if the identifier is a valid identifier.
                if isinstance(identifier, Identifier):
                    # Return the real name of the identifier.
                    yield identifier.get_real_name()


def append_view_to_tables(sql, tables, add_view_suffix=False):
    """Append the view suffix to the tables in the SQL.

    Args:
        sql (str): SQL statement to append the suffix to.
        tables (list): List of tables to append the suffix to.
        add_view_suffix (bool, optional): If True, append the suffix. Defaults to False.

    Returns:
        str: Updated SQL statement.
    """
    updated_sql = sql
    if add_view_suffix:
        for table in tables:
            updated_sql = updated_sql.replace(table, f"{table}_view")
    return updated_sql


def extract_tables(sql):
    parsed = sqlparse.parse(sql)[0]
    token_stream = extract_table_identifiers(parsed.tokens)
    table_set = set()

    for token in token_stream:
        if token != "AS":
            table_set.add(token)

    return list(table_set)




 
# Append _view to table names
#add_view_suffix = st.checkbox("Append '_view' to table names")


# Convert SQL dialect
if st.button("Mask"):
#if sql:  
    st.write("Encrpyting your SQL Code...")
    list_of_fields = get_identifiers(sql)
    masked_sql, word_map = sql_masking(list_of_fields, sql)
    st.code(masked_sql)

# st.write("Enter text into field 1 - this will hide field 2")
# text_field_1 = st.text_input("Field 1")

# if not text_field_1:
#     st.write("Enter text into field 2 - this will be hidden if field 1 has text")
#     text_field_2 = st.text_input("Field 2")

# if not sql:
#     st.write("Encrpyting your SQL Code...")
#     text_field_2 = st.text_input("Field 2")

if st.button("Dialectify"):
    st.write(f"Converting your SQL Code from {from_sql} to {to_sql}...")
    list_of_fields = get_identifiers(sql)
    masked_sql, word_map = sql_masking(list_of_fields, sql)
    masked_converted_sql = sql_dialectify(from_sql, to_sql, masked_sql, max_tokens, model_choice, temperature)

    demasked_sql = demasking(word_map, masked_converted_sql)

    # tables = extract_tables(demasked_sql)
   
    # updated_sql = append_view_to_tables(demasked_sql, tables, add_view_suffix)
    
    # st.subheader("Original SQL:")
    # st.code(sql)
    st.subheader("Updated SQL:")
    formatted_sql = sqlparse.format(demasked_sql, reindent=True, keyword_case='upper')
    
    st.code(formatted_sql, language="sql")
    st.write("Done! This was fun, now give me another one!")
    


    



