import os
import shutil
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

import mysql.connector as mysql
import pymysql
from sqlalchemy import create_engine
host=os.getenv('MYSQL_SERVER')
database=os.getenv('MYSQL_DATABASE')
user=os.getenv('MYSQL_USER')
password=os.getenv('MYSQL_PASSWORD')


def create_dbengine():
    ### Create SQLAlchmey engine
    # Create the engine to connect to the MySQL database
    connect_args={'ssl':{'fake_flag_to_enable_tls': True}}
    return create_engine(f'mysql+pymysql://{user}:{password}@{host}:3306/{database}', connect_args=connect_args)

def create_dbconnection():
    try:
        db = mysql.connect(host=host, user=user, password=password, database=database)
        return db, db.cursor()
    except mysql.Error as e:
        if e.errno == mysql.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
            return None
        elif e.errno == mysql.ER_BAD_DB_ERROR:
            print("Database does not exist")
            return None
        else:
            print(e)
            return None
    else:
        return None
    
def execute_dbquery(query, db=None, cursor=None):
    # If no db or cursor is provided, connect to the database
    if db is None or cursor is None:
        db, cursor = create_dbconnection()

    try:
        cursor.execute(query)
        db.commit()
        # time.sleep(5)
    except mysql.Error as e:
        print(f"Failed creating database with query: {query} - Error: {e}")
    
def write_sql_file(sql_file, sql_query):
    with open(f'sql_generated/{sql_file}.sql', 'w') as f:
        f.write(sql_query)

def dtype_by_format(format, dtype, width):
    dataType = None
    ###
    ### If the column is an alpha column, set the data type to VARCHAR for the specified field width
    ###
    if format.upper() == 'A':
        dataType = f'VARCHAR({width})'

    ###
    ### If the column is not-alpha, determine the numeric data type and field width
    ###
    else:
        # Check to see if data type is integer or float, but looking at how the data would import
        if dtype == 'float':
            if width > 8:
                dataType = 'DOUBLE'
            else:
                dataType = 'FLOAT'
        else:
            if width > 8:
                dataType = 'BIGINT'
            elif width > 6:
                dataType = 'INT'
            elif width in [5, 6]:
                dataType = 'MEDIUMINT'
            elif width in [3, 4]:
                dataType = 'SMALLINT'
            elif width in [1, 2]:
                dataType = 'TINYINT'
            else:
                dataType = 'INT'
    ###
    ### If the data type is still None, set it to TEXT
    ###
    if dataType == None:
         print(f'**WARNING**: {row.varname_new} has no data type, setting to TEXT')
         dataType = 'TEXT'
    ###
    ### Return the data type
    ###
    return dataType

def dtype_by_datatype(dtype, length):
    if dtype == 'int64':
        return 'INTEGER'
    elif dtype == 'float64':
        return 'REAL'
    elif dtype == 'object':
        return 'TEXT'
    else:
        return 'TEXT'
    
###
### Create column names from the column descriptions using GPT-3.5-turbo OpenAI API
###
def create_name_from_description(titles):
    titleList = ""
    for title in titles:
        title = title.replace(",", "").replace("'", "").replace(".", "").lower()
        titleList += f'[{title}],'
    message = {"role": "user", "content": f"{titleList}"}
    messages = [
        {
        "role": "system",
        "content": """
            Produce a list of readable database column names using the list of Column Descriptions provided.
            Column Names should NOT start with a number, be MySQL reserved word or Python keyword. 
            All special characters are to be replaced with an underscore and the Column Name should be all lowercase. 
            Make sure Column Names are readable, descriptive, concise, consistent, unique, and less than 30 characters including underscores.
            Return one entry for each Column Name submitted by the user.
            DO NOT return a numbered list.
            Substitute common words in the Column Descriptions with abbreviations such as:
                'inst' replaces 'institution'
                'id' replaces 'identification'
                'class' replaces 'classification'
                'cd' replaces 'code'
                'org' replaces 'organization'
                'loc' replaces 'location'
                'cat' replaces 'category'
                'url' replaces 'web address'
                'url' replaces 'website address'
                'rpt' replaces 'report'
            Returned Column Name(s) should be in square brackets."""
        },
        {
        "role": "user",
        "content": "[Unique identification number of the institution]"
        },
        {
        "role": "assistant",
        "content": "[uid]"
        },
        {
        "role": "user",
        "content": "[Institution (entity) name], [Institution name alias]"
        },
        {
        "role": "assistant",
        "content": "[inst_name], [inst_alias]"
        },
        {
        "role": "user",
        "content": """[Title of chief administrator], 
                [Disability Services Web Address], 
                [Institution's internet web address], 
                [Office of Postsecondary Education (OPE) ID Number], 
                [Sector of institution]"""
        },
        {
        "role": "assistant",
        "content": "[title_chief_administrator], [disability_url], [inst_url], [ope_id], [sector]"
        }
    ]
    messages.append(message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0]['message']['content'].replace("[", "").replace("]", "").replace(" ", "").split(",")

###
### Get list of dictionary files already processed through OpenAI API
###   - This is to avoid re-processing the same files again
###
def return_list_of_processed_tables():
    directory_path = 'dictionary/'
    file_pattern = '*.new.xlsx'

    files_list = glob.glob(directory_path + '/' + file_pattern)
    new_files_list = []
    for each_file in files_list:
        new_files_list.append(each_file.replace(directory_path, '').replace('.new.xlsx', '').upper())

    return new_files_list