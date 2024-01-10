import re
import os
import pandas as pd
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
# from tqdm import tqdm

#openai.api_key = os.getenv('OPENAI_API_KEY')

# Declare word_map and masked_converted_sql as global variables
word_map = {}
masked_converted_sql = ""


# Set the app title
st.title("Dialectify SQL")
st.markdown("#### Switch between SQL dialects")
st.markdown("###### Dialectify, transforms your SQL scripts, into any SQL dialect of another platform. As it does, you have the option to mask your secret table and field names, or even add the typical '_view' suffix used in data warehousing implementations. Github Copilot does not have SQL to SQL conversion!")

# Create the input boxes for the SQL code and the SQL dialects
openai.api_key = st.text_input("Enter your OPENAI API Key:", type="password")
sql = st.text_area('Enter SQL to Convert', height=400, key='sql', help="Enter SQL to convert")

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


# Create a sidebar in Streamlit
st.sidebar.title("Select Model & Temperature")


model_choice = st.sidebar.radio("Model:", ["gpt-4", "gpt-3.5-turbo"])

#max_tokens = st.sidebar.selectbox("Enter Max Tokens", [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192], index=0) 
st.sidebar.markdown("Models are set to return max allowed tokens. \n\n (max allowed tokens = inputted tokens + returned tokens) \n\n GPT-4 has a maximum token limit of 8,192 tokens (equivalent to ~6000 words), whereas GPT-3.5's 4,000 tokens (equivalent to 3,125 words).")


temperature = st.sidebar.selectbox("Temperature:", [0, 0.1, 0.2, 0.3, 0.9], index=0)

def sql_dialectify(from_sql, to_sql, original_sql, model_choice=model_choice, temperature=temperature, mask_fields=False, identifiers=None):
    if from_sql == to_sql:
        return "Pick a different dialect then the original plz"
    word_map = {}  # Initialize word_map
    # If mask_fields is True, mask the fields using the provided sql_masking function
    if mask_fields and identifiers:
        masked_sql, word_map = sql_masking(identifiers, original_sql)
    else:
        masked_sql = original_sql

    response = openai.ChatCompletion.create(
        model=model_choice,
        temperature=temperature,

        messages=[
           {"role": "system", "content": 'Act as CODEX ("COding DEsign eXpert"), an expert coder with proficiency in SQL programming language.'},
            {"role": "system", "content": 'You are proficient in SQL dialects, with a focus on high accuracy dialect to dialect conversions.'},
            {"role": "system", "content": 'Your task is to convert a specific SQL script from one SQL dialect to another SQL dialect while maintaining the functionality and integrity of the original script.'},
            {"role": "system", "content": f'The source SQL dialect is "{from_sql}", and the target SQL dialect is "{to_sql}". Your goal is to perform a precise SQL dialect conversion while addressing any incompatibilities or differences.'},
            {"role": "system", "content": 'Always follow the coding best practices by writing clean, modular code with proper security measures and leveraging design patterns.'},
            {"role": "system", "content": f'Let''s think step by step. First, you will identify the differences between the {from_sql} and {to_sql} dialects. Then, you will convert the SQL code from {from_sql} to {to_sql}.'},
            {"role": "system", "content": f'You will identify and address differences in data types and functions between the {from_sql} and {to_sql} dialects. For data types or functions without a direct equivalent, choose the most suitable alternative'},
            {"role": "system", "content": 'When converting to Snowflake SQL do not use double quotes for column and table names instead of square brackets that is used in Transact-SQL. Instead of square brackets, use nothing.'}, 
            {"role": "system", "content": 'Output results in a sql query format. Transform the column and table names of the converted SQL query to UPPERCASE. Kindly only provide the converted SQL query and nothing else in your response.  '},
            {"role": "user", "content": f'Convert the SQL dialect of the following SQL query in backticks from "{from_sql}" to "{to_sql}" while ensuring the highest level of accuracy in maintaining the original functionality: ```\n\n{masked_sql}```\n\n'}
        
        ]

    )

    converted_sql = response.choices[0].message.content

    # If mask_fields is True, replace the masked words with the original identifiers
    if mask_fields and identifiers:
        for original, masked in word_map.items():
            converted_sql = converted_sql.replace(masked, original)
    # # Use tqdm to display progress bar
    # for i in tqdm(range(100)):
    #     pass

    return converted_sql


# get the list of tables in a query
#@st.cache_data
def tables_in_query(sql_str):

    # remove the /* */ comments
    q = re.sub(r"/\*[^*]*\*+(?:[^*/][^*]*\*+)*/", "", sql_str)

    # remove whole line -- and # comments
    lines = [line for line in q.splitlines() if not re.match("^\s*(--|#)", line)]

    # remove trailing -- and # comments
    q = " ".join([re.split("--|#", line)[0] for line in lines])

    # split on blanks, parens and semicolons
    tokens = re.split(r"[\s)(;]+", q)

    # scan the tokens. if we see a FROM or JOIN, we set the get_next
    # flag, and grab the next one (unless it's SELECT).

    result = set()
    get_next = False
    for tok in tokens:
        if get_next:
            if tok.lower() not in ["", "select"]:
                result.add(tok)
            get_next = False
        get_next = tok.lower() in ["from", "join"]

    return result

# Extract tables for db views

if st.button("Extract tables", use_container_width=True):
    # Extract tables from SQL code
    st.write(f"Here are the table names. Check if the view definitions already exist in the db...")
    tables = tables_in_query(sql)

    # Display tables in a data frame but do not display the index, remove the number on the site and make the column name nothing
    df_of_tables = pd.DataFrame(list(tables), columns=[''])
    df_of_tables.index.name = ''

    # Save the data frame in the session state
    st.session_state.table_df = df_of_tables

    # Display the data frame
    st.table(df_of_tables)
    
st.divider()
# From and To sql dialect choices
from_sql = st.selectbox("From SQL:", ["Transact-SQL", "MySQL", "PL/SQL", "PL/pgSQL", "SQLite", "Snowflake"])
to_sql = st.selectbox("To SQL:", ["Transact-SQL", "MySQL", "PL/SQL", "PL/pgSQL", "SQLite", "Snowflake"])

#mask fields or not
mask_fields = st.checkbox("Mask all fields including table/view names before sending the query to OpenAI")
# Append _view to table names
add_view_suffix = st.checkbox ("Append '_view' to table names after the conversion")


if st.button("Dialectify", use_container_width=True):
    with st.spinner(f"Converting SQL code from {from_sql} to {to_sql}..."):      
        converted_sql = sql_dialectify(from_sql, to_sql, sql, model_choice, temperature, mask_fields)

        #Add view suffix to table names if the property is checked
        if add_view_suffix:
            tables = tables_in_query(converted_sql)
            for table in tables:
                converted_sql = converted_sql.replace(table, f"{table}_VIEW")
            st.subheader(f'Your SQL code is converted from {from_sql} to {to_sql}. Validate the output!')
    st.code(converted_sql)
   
   
    
