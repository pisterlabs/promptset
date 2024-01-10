import openai
import os
from dotenv import load_dotenv
import json
import logging
from datetime import datetime

load_dotenv()  # load environment variables from .env file

openai.api_key = os.getenv('OPENAI_API_KEY')  # get OpenAI key from environment variables
port = os.getenv('getSkills_PORT', 5001)
debug = os.getenv('getSkills_DEBUG', True)
log_level = os.getenv('getSkills_LOG_LEVEL', 'WARNING').upper()

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

logger.info('''
            Error levels:
            * DEBUG
            * INFO
            * WARNING
            * ERROR
            * CRITICAL''')

role = "Agile Coach"
logger.info(f'Role: {role}')

competency = "Emergent"
logger.info(f'Competency: {competency}')

current_answer = ""
logger.info(f'Current answer: {current_answer}')


prompt = f'''
    As an expert HR consultant, I am evaluating the competencies necessary for the role of "{role}" at the "{competency}" level, using a competency scale of emergent, competent, expert, and lead. 

    Please generate a comprehensive overview that includes skills, knowledge, key deliverables, and personal attributes contributing to success in this role. 

    Present the information as a JSON object.

    Within the required and recommended array, each object should represent a key competency required for the role. 
    Each competency should be broken down into sub-elements. 
                  
    Ensure the "Skills" array contains specific skills associated with each competency.'''
logger.info(f'Prompt: {prompt}')

roleformat = {
        "Role": "Role Name",
        "Level": "Competency Level",
        "Description": "Role Description",
        "Required": [
              {
                  "Name": "Element Name",
                    "Level": "Element Level",
                    "Description": "Element Description",
                    "Skills": [
                      "Skill 1",
                      "Skill 2",
                      "etc"
                    ],
                    "Knowledge": [
                        "Knowledge 1",
                        "Knowledge 2",
                        "etc"
                    ],
                    "Deliverables": [
                        "Deliverable 1",
                        "Deliverable 2",
                        "etc"
                    ],
                    "Attributes": [
                        "Attribute 1",
                        "Attribute 2",
                        "etc"
                    ]
              },
              "etc"
          ],
        "Recommended": [
              {
                    "Name": "Element Name",
                    "Level": "Element Level",
                    "Description": "Element Description",
                    "Skills": [
                      "Skill 1",
                      "Skill 2",
                      "etc"
                    ],
                    "Knowledge": [
                        "Knowledge 1",
                        "Knowledge 2",
                        "etc"
                    ],
                    "Deliverables": [
                        "Deliverable 1",
                        "Deliverable 2",
                        "etc"
                    ],
                    "Attributes": [
                        "Attribute 1",
                        "Attribute 2",
                        "etc"
                    ]
              },
              "etc"
          ]
      }
logger.info(f'Role format: {roleformat}')

expertformat = {"experts":[{"Name": "Expert Name", "Description": "Expert Description"}]}
logger.info(f'Expert format: {expertformat}')

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

    if response['choices'] and response['choices'][0]['message']:
        logger.info(response['choices'][0]['message']['content'].strip())
        return response['choices'][0]['message']['content'].strip()

def get_skills():
    
    prompttext = prompt + '''
      Use the following format for the JSON object:
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

    if response['choices'] and response['choices'][0]['message']:
      logger.info(response['choices'][0]['message']['content'].strip())
      return response['choices'][0]['message']['content'].strip()
    
def get_review(data):
    prompttext = f'''Pretend you are {data} reviewing the output of the following task:
    
    {prompt}

    The json object to review is:
    
    {current_answer}
    
    Firstly, review the list for completeness, and add any elements you think are missing.

    Then review each element, for each element, review the list of skills, knowledge, deliverables, attributes to make sure they are appropriate for the role and level.
    Then add your assessment of that element to the score array, add your name, a score from 1 to 100, and a comment, noting if you added items, on how you scored the element.
        
    Only return the updated json object.'''
    logger.info(prompttext)

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
        {
            "role": "assistant",
            "content": prompttext
        }
      ]
    )
    logger.debug(response)

    if response['choices'] and response['choices'][0]['message']:
      logger.info(response['choices'][0]['message']['content'].strip()) 
      return response['choices'][0]['message']['content'].strip()

if __name__ == '__main__':
    current_answer = get_skills()
    logger.info(f'Current answer: {current_answer}')

    # Now we add the scoring element to each element in the json object
    json_obj = json.loads(current_answer)

    # Add "Score" field to each element in "Required" and "Recommended" fields
    for item in json_obj['Required']:
        item['Score'] = []

    for item in json_obj['Recommended']:
        item['Score'] = []

    # Convert Python object back to JSON string
    current_answer = json.dumps(json_obj, indent=2)
    logger.debug(current_answer)

    experts = get_experts()
    logger.info(f'Experts: {experts}')

    data = json.loads(experts)
    for expert in data['experts']:
        current_answer = get_review(expert['Name'])
        logger.info(current_answer)
        
    print('''Final Answer:
          
          ''' + current_answer)