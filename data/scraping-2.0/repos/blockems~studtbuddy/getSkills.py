#OS imports
import os
import json
import logging
from datetime import datetime

#Env Mgt
from dotenv import load_dotenv

#Web Page
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

#JSON Structures
from pydantic import BaseModel
from typing import List

#AI Client
import openai

#Database Access
import sqlite3

openai.api_key = os.getenv('OPENAI_API_KEY')  # get OpenAI key from environment variables
port = os.getenv('getSkills_PORT', 5001)
debug = os.getenv('getSkills_DEBUG', True)
log_level = os.getenv('getSkills_LOG_LEVEL', 'WARNING').upper()
data_dir = os.getenv('DATA_DIRECTORY', './data/roles')
data_conn = os.getenv('DATA_CONNECTION_STRING', './data/database.db')

# Define the directory path for the JSON data files
directory_path = data_dir

# Create a connection to the SQLite database
conn = sqlite3.connect(data_conn)

#Set up logging
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')

logger = logging.getLogger(__name__)
logger.setLevel(numeric_level)

# create formatter
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')

# Create a FileHandler for logging data to a file
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'log_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log')
fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)  # Set the formatter for the file handler
logger.addHandler(fh)  # Add the file handler to the logger

logger.info(f'port: {port}')
logger.info(f'debug: {debug}')
logger.info(f'log_level: {log_level}')
logger.info(f'numeric_level: {numeric_level}')
logger.info(f'Data Directory: {data_dir}')
logger.info(f'Data Connection String: {data_conn}')

current_answer = ""
competency = ""
role = ""
prompt = ""

#Now define our json object:
class Skill(BaseModel):
    Name: str
    Level: str
    Description: str

class Reviewer(BaseModel):
    Name: str
    Description: str

class RoleDefinition(BaseModel):
    Role: str
    Level: str
    Description: str
    Required: List[Skill]
    Recommended: List[Skill]
    Reviewers: List[Reviewer]

schema = Skill.schema()

class Expert(BaseModel):
    Name: str
    Description: str

class ExpertsModel(BaseModel):
    experts: List[Expert]

expert = Expert.schema()

#Make an API
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def get_experts():
    prompttext = '''I want a response to the following question:
    
    ''' + prompt + '''

        Name 3 world-class experts (past or present) who would be great at answering this?
        Don't answer the question yet. Just name the experts.
        Please return the results as a json object with the following structure:    
        '''
    
    prompttext = prompttext + json.dumps(expertformat)
    logger.info(prompttext)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "assistant",
            "content": prompttext
          }
        ]
    )
    logger.debug(response)

    if response['choices'] and response['choices'][0]['message']: # type: ignore
        logger.info(response['choices'][0]['message']['content'].strip()) # type: ignore
        return response['choices'][0]['message']['content'].strip() # type: ignore

def get_skills():
    
    prompttext = prompt + '''
      
      The JSON object is as follows:
      ''' + json.dumps(roleformat)
    logger.info(prompttext)

    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {
            "role": "assistant",
            "content": prompttext
        }
      ]
    )
    logger.debug(response)

    if response['choices'] and response['choices'][0]['message']: # type: ignore
      logger.info(response['choices'][0]['message']['content'].strip()) # type: ignore
      return response['choices'][0]['message']['content'].strip() # type: ignore
    
def get_review(data):
    prompttext = f'''
    Your role is {data} reviewing the output of a task. In this capacity, you're a critical yet supportive reviewer aiming to enhance the quality of the work.
        
    The task you are reviewing involves the following:
    
    {prompt}

    The JSON object is as follows:
    
    {current_answer}
    
    As {data}, your task is to review the list methodically for completeness. 
    Then, add your name and a feedback to the reviewers array with comprehensive feedback on how the list could be improved. 
    This includes suggesting additional skills and knowledge, as well as improvements to existing skills and descriptions. 

    The final output should be the JSON object, with your feedback in the reviewers array.'''
    logger.info(prompttext)

    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {
            "role": "assistant",
            "content": prompttext
        }
      ]
    )
    logger.debug(response)

    if response['choices'] and response['choices'][0]['message']: # type: ignore
      logger.info(response['choices'][0]['message']['content'].strip()) # type: ignore
      return response['choices'][0]['message']['content'].strip() # type: ignore

def get_final_answer():
    prompttext = f'''role:
    You are a HR consultant in a large financial institution.

    Task:
    You are reviewing and updating the skills and competencies required to be successful in the IT part of your organization. for the role of "{role}" at the level of "{competency}"
    Taking the feedback from the experts that have added thier comments to the review section of the current list.
    Think through each element step by step, and where nescesary, consolodate the feedback from the experts.

    Add your review comments to the reviewers section of the json object.
    
    Finally review and update the list. 
    
    Only return the updated json object.
    
    The current list is:
    {current_answer}
    '''

    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {
            "role": "assistant",
            "content": prompttext
        }
      ]
    )
    logger.debug(response)

    if response['choices'] and response['choices'][0]['message']: # type: ignore
      logger.info(response['choices'][0]['message']['content'].strip()) # type: ignore
      return response['choices'][0]['message']['content'].strip() # type: ignore

#Set up an API end point
@app.route('/getSkills', methods=['POST'])
@cross_origin()
def skills():
    data = request.get_json()
    role = data.get('role')
    logger.info(f'Role: {role}')
    competency = data.get('competence')
    logger.info(f'Competency: {competency}')

    prompt = f'''
    Role:
    You are a HR consultant in a large financial institution.

    Background:
    Your task involves understanding the necessary skills and competencies for success in the IT department of your organization. The role requires the appropriate level of proficiency with modern software construction methodologies, in addition to thorough familiarity with the business and technical aspects of banking, and regulatory requirements.

    Task:
    Given a competency scale of emergent, competent, expert, and lead, break down the skills and knowledge necessary for an "Emergent" level "Agile Coach". Present the information in the form of a JSON object.

    Output:
    The task's output is a JSON object which outlines the required and recommended skills for an "{competency}" level "{role}".
    '''
    logger.info(f'Prompt: {prompt}')

    #Get the first qttempt and an answer
    current_answer = get_skills()    
    if current_answer:
      try:
          json_obj = json.loads(current_answer)
          #Add the reviewers section
          json_obj['Reviewers'] = [{"Name": "Reviewer Name", "Description": "Reviewer Description" }]

      except json.JSONDecodeError:
        logger.error(f"Invalid JSON string: {current_answer}")
        logger.info(f'Current answer: {current_answer}')
    
    #Get the experts to review the answer
    experts = get_experts()
    if experts:
      try:
          json_obj = json.loads(experts)
          logger.info(f'Experts: {experts}')
          #Add the reviewers section
          json_obj['Reviewers'] = [{"Name": "Reviewer Name", "Description": "Reviewer Description" }]

      except json.JSONDecodeError:
        logger.error(f"Invalid JSON string: {experts}")
        logger.info(f'Current answer: {experts}')
    
    #Now update the working JSON object to send to the AI 
    current_answer = json.dumps(json_obj)
    logger.info(f'Current answer: {current_answer}') 
    
    # Now we go and get the reviewers comments
    data = json.loads(experts)
    for expert in data['experts']:
        current_answer = get_review(expert['Name'])
        logger.info(current_answer)

    current_answer = get_final_answer()
    logger.info(current_answer)

    return jsonify(current_answer)

if __name__ == '__main__':
    app.run(debug=debug, port=port)