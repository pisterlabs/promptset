import openai
import os
import re
#get the api key from .env file
from dotenv import load_dotenv
load_dotenv()

class mongo_make:

    def __init__(self, input):
        self.input = input
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.context = """
            Accurately identify different aspects of the input, like collection names, fields, and their data types.
            Translate these aspects into structured Python code for database operations using MongoDB.
            Maintain readability and best practices in code generation.
            Here's an optimized version of the prompt:

            Context Prompt for Chatbot:

            As a software developer, your task is to transform a program description into Python code for managing a MongoDB database. The code should include:

            Importing necessary modules.
            Defining the MongoDB URI and a function to connect to the database.
            Creating functions to handle database operations such as creating data entries and adding collections with specified schemas.
            Determining the database schema based on the provided program description.
            Input Example:

            The input will describe a database with multiple collections. Each collection will have its fields and data types specified. For example:

            Database Schema:
            Collection 1: Requests
            Fields: Request Id, Request Title, Request Description, Request Created Date, Request Status
            Collection 2: Users
            Fields: User Id, User Type (worker/IT), Name
            Expected Output:

            Based on the input, generate Python code that includes:

            Import statements for MongoDB and other necessary modules.
            A MongoDB URI string.
            Functions for connecting to the database and performing CRUD operations.
            Database schemas for each collection as Python dictionaries, with field names and data types.
            The code should establish a database connection, add collections with the defined schemas, and output confirmation of these actions.
            The output should be well-formatted, syntactically correct Python code, ready for execution.
            """
        self.template = """from pymongo import MongoClient
            from datetime import datetime
            from typing import Dict

            # MongoDB URI
            mongo_uri = "mongodb://localhost:27017/"

            # Function to connect to the database
            def connect_db(uri: str, db_name: str):
                client = MongoClient(uri)
                db = client[db_name]
                return db

            # Function to create a data entry in a collection
            def create_data_entry(db, collection_name: str, data: Dict):
                collection = db[collection_name]
                return collection.insert_one(data).inserted_id

            # Function to add a collection with a specific schema
            def add_table(db, table_name: str, data_schema: Dict):
                collection = db[table_name]
                return collection.insert_one(data_schema).inserted_id

            # Database name
            db_name = "userDB"
            db = connect_db(mongo_uri, db_name)

            # Collection and schema for Requests
            requests_table = 'requests'
            requests_schema = {
                "Request_ID": int,
                "Request_Title": str,
                "Request_Description": str,
                "Created_Date": datetime,
                "Request_Status": str
            }
            requests_table_id = add_table(db, requests_table, requests_schema)

            # Collection and schema for Users
            users_table = 'users'
            users_schema = {
                "User_ID": int,
                "User_Type": str, # worker/IT
                "User_Name": str
            }
            users_table_id = add_table(db, users_table, users_schema)

            # Outputting the created tables and their schemas
            print(f"Requests table created with ID: {requests_table_id}")
            print(f"Requests table schema: {requests_schema}")
            print(f"Users table created with ID: {users_table_id}")
            print(f"Users table schema: {users_schema}")

            """
    def interact(self):
        response = openai.chat.completions.create(
        model= "gpt-4",
        messages=[
            {"role": "system", "content": self.context,
            "role": "user", "content": self.input,}])
        return response.choices[0].message.content

        
    def run_interaction(self):

            self.input = str(self.input + "Generate the code following this fomrat: " + self.template)
            return self.interact()

    def extract_code_from_markdown(markdown_text):
        # Regular expression pattern for code blocks
        pattern = r"```(.*?)```"

        # Find all non-overlapping matches in the markdown text
        matches = re.findall(pattern, markdown_text, re.DOTALL)

        # Process each match to maintain its original formatting
        formatted_code_blocks = []
        for match in matches:
            # Add each code block as a single string, preserving whitespace and newlines
            formatted_code = '\n'.join(match.splitlines())
            formatted_code_blocks.append(formatted_code)

        return formatted_code_blocks

    def to_python_file(self, file_name="mongo_creation_script.py"):

        code = self.run_chain()
        code_str = code[0]
        # Remove the initial 'Python\n    ' and adjust indentation
        formatted_code = '\n'.join(line[4:] for line in code_str.split('\n')[1:])

        # Write the formatted code to a file
        with open(file_name, 'w') as file:
            file.write(formatted_code)

        

    def run_chain(self):

        # call the llm
        llm_output = self.run_interaction()

        # extract the code from the llm output
        code = self.extract_code_from_markdown(llm_output)

        # return the code
        return code

#let's run it
input = """Database Schema: Collection 1: Requests Fields:

Request Id
Request Title
Request Description
Request Created Date
Request Status
Collection 2: Users Fields:

User Id
User Type (worker/IT)
Name
User Types and Access: Workers: -Can view all requests -Can submit requests -No access to IT requests

IT Department: -Can view all requests -Can mark requests as pending, complete, or underway -Can delete requests

Pages and Permissions: Submitter View:

Accessible to all workers
Displays a list of all requests
Has input fields to submit new requests
Read-only access
IT View:

Accessible to IT department
Displays a list of all requests
Has input fields to mark requests as pending, complete, or underway
Edit and delete access"""

mongo_maker = mongo_make(input)
mongo_maker.to_python_file()
