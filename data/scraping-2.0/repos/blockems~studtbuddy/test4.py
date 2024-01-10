#OS imports
import os
import json
import logging
from datetime import datetime

#Env Mgt
from dotenv import load_dotenv

#JSON Structures
from pydantic import BaseModel
from typing import List

#AI Client
import openai

#Database Access
import sqlite3

#Load env variables:
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')  # get OpenAI key from environment variables
port = os.getenv('getSkills_PORT', 5001)
debug = os.getenv('getSkills_DEBUG', True)
#log_level = os.getenv('getSkills_LOG_LEVEL', 'WARNING').upper()
log_level = 'DEBUG'
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

role = "Agile Coach"
competency = "Emergent"
current_answer = ""
current_json = ""
prompt = ""

#Now define our json objects:
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

skill_schema = {
    "name" : "get_skills",
    "description" : "Get the skills for the role",
    "parameters" : RoleDefinition.schema(),
    "return_type" : "json" 
    }

class Expert(BaseModel):
    Name: str
    Description: str

class ExpertsModel(BaseModel):
    experts: List[Expert]

expert_schema = {
    "name" : "get_experts",
    "description" : "Get a list of experts for answering this question",
    "parameters" : ExpertsModel.schema(),
    "return_type" : "json" 
    } 
    
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

    if response['choices'] and response['choices'][0]['message']: #type: ignore
      logger.info(response['choices'][0]['message']['content'].strip())  #type: ignore
      return response['choices'][0]['message']['content'].strip() #type: ignore

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

    if response['choices'] and response['choices'][0]['message']: #type: ignore
      logger.info(response['choices'][0]['message']['content'].strip()) #type: ignore
      return response['choices'][0]['message']['content'].strip() #type: ignore

def get_skills():
    
    logger.debug(f'Prompt: {prompt}')
    logger.debug(f'Skill schema: {skill_schema}')
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {
            "role": "assistant",
            "content": f'{prompt} Dont populate the reviewers list at this stage.'
        }
      ],
      functions = [skill_schema],
      function_call = {"name":"get_skills"}
    )
    logger.debug(f'response:{response}')

    #if we have a response
    if response['choices'] and response['choices'][0]['message']: #type: ignore
      return_json = json.loads(response.choices[0]["message"]["function_call"]["arguments"]) #type: ignore
      logger.info(f'First Answer run: {return_json}')
      return return_json

def get_experts():
    prompttext = '''I want a response to the following question:
    
    ''' + prompt + '''

        Name 3 world-class experts (past or present) who would be great at answering this?
        Don't answer the question yet. Just name the experts.
        '''
    logger.info(prompttext)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "assistant",
            "content": prompttext
          }
        ],
      functions = [expert_schema],
      function_call = {"name":"get_experts"}
    )
    logger.debug(response)

    if response['choices'] and response['choices'][0]['message']: #type: ignore
      return_json = json.loads(response.choices[0]["message"]["function_call"]["arguments"]) #type: ignore
      logger.info(f'List of Experts: {return_json}')
      return return_json

if __name__ == '__main__':
    prompt = f'''
    Role:
    You are a HR consultant in a large financial institution.

    Background:
    Your task involves understanding the necessary skills and competencies for success in the IT department of your organization. The role requires the appropriate level of proficiency with modern software construction methodologies, in addition to thorough familiarity with the business and technical aspects of banking, and regulatory requirements.

    Task:
    Given a competency scale of emergent, competent, expert, and lead, break down the skills and knowledge necessary for an "Emergent" level "Agile Coach". Present the information in the form of a JSON object.

    Output:
    The task's output is a JSON object which outlines the required and recommended skills for an "{competency}" level "{role}".
    The JSON should have a "required" array where you list the skills.

    '''
    #Get the first answer
    current_json = get_skills()
    #get some experts to delibarate
    experts = get_experts()
        
    # Now we go and get the reviewers comments
    data = json.loads(experts)
    for expert in data['experts']:
        current_answer = get_review(expert['Name'])
        logger.info(current_answer)

    #current_answer = get_final_answer()
    #logger.info(current_answer)