import os
from pdfminer.high_level import extract_text
import openai
import sqlite3
import json
openai.api_key = "Insert API Key"

# A function to extract text from pdfs
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Path to the folder containing resumes
resume_folder = "path to folder containing resumes"

# List all files in the folder
files = os.listdir(resume_folder)

### A for loop to go through all the resumes in a folder ###
for file in files:
    # Check if the file is a resume
    if file.endswith(".pdf") or file.endswith(".docx") or file.endswith(".txt"):
        # Process the resume file and extract the text
        resume_path = os.path.join(resume_folder, file)
        text = extract_text_from_pdf(resume_path)

        ### Beginning of call to GPT ###
        ## Defining the role of the AI ##
        messages = [{'role':'system', 'content': 'You are a recruitment assistant for a multi-million dollar company that parses resumes to extract key information'}]

        ## Defining what we want GPT to do ##
        prompt = f"Given a pool of resumes, your goal is to extract and list for me all of thier skills (technical and soft),current location, name, phone number, email, educational background, certifications and total years of work experience. ###" \
                 f"Give your answer in a dictionary format with the following keys (case sensitve): [skills, location, name, phone_number, email, educational_background, certifications and total_years_of_work_experience]. Your response to each of these will be the values. You can choose to organise the values for each key in a list format " \
                 f"###" \
                 f"use double quotes for keys and string values as I should be readily able to convert this to an actual dictionary by using the json.loads function" \
                 f"Give the value for the key 'total years of work experience' as a single number formatted as a string" \
                 f"Resume: ###{text}###"
        messages.append({'role': 'user', 'content': prompt})

        ## making a call to GPT using the following function ##

        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages = messages

        )
        # the return statement of the GPT model
        response = completion.choices[0].message.content

        # Next I decided to store the data permanently in a database to prevent calling GPT again for the same data.

        ### SQL code begins here ###
        conn = sqlite3.connect('resume_database.db')
        cursor = conn.cursor()
        # cursor.execute('DROP TABLE IF EXISTS my_table') # need to delete this before deployment

        # creates a table with the specified column names and their data type
        cursor.execute('''CREATE TABLE IF NOT EXISTS my_table (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT,
                          skills TEXT,
                          location TEXT,
                          phone_number TEXT,
                          email TEXT,
                          educational_background TEXT,
                          certifications TEXT,
                          total_years_of_work_experience TEXT)''')

        # One major problem is GPT returns text values. It is in string format. Need to convert it ourselves into dict type
        item = json.loads(response)

        # temporary placeholder for values that will go in each column of the table
        insert_sql = '''INSERT INTO my_table (name, skills, location, phone_number,
                        email, educational_background, certifications, total_years_of_work_experience)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''

        # used json.dumps to convert the type into TEXT as this is the format
        # sql database accepts. Can't accept a list which is the original output
        cursor.execute(insert_sql, (
                json.dumps(item['name']), json.dumps(item['skills']),
                json.dumps(item['location']), json.dumps(item['phone_number']),
                json.dumps(item['email']),
                json.dumps(item['educational_background']),
                json.dumps(item['certifications']),
                item['total_years_of_work_experience']))

        conn.commit()
        conn.close()

        # at the end of this, one row of our table is populated with a specific
        # resume's features. Loop will continue till the last file in the folder


