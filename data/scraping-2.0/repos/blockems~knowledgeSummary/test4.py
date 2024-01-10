#OS imports
import os
import json
from datetime import datetime

#Env Mgt
from dotenv import load_dotenv

#JSON Structures
from pydantic import BaseModel
from typing import List
from datetime import date

#AI Client
import openai

#PDF tools
import PyPDF2

#File/shell tools
import shutil

#Token counter for modeling size to stuff into GPT
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4-0613")

#Bring in the utils
from utils import init_logging, init_db

#Load env variables:
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')  # get OpenAI key from environment variables
debug = os.getenv('getSkills_DEBUG', True)
log_level = 'DEBUG'
data_conn = os.getenv('DATA_CONNECTION_STRING', './data/database.db')
data_source= os.getenv('DATA_SOURCE_DIR', './source')
data_processed= os.getenv('DATA_PROCESSED_DIR', './source/processed')
data_pre_processed= os.getenv('DATA_PRE_PROCESSED_DIR', './source/preprocessed')

# Initialize logging
logger = init_logging(log_level)

logger.info(f'debug: {debug}')
logger.info(f'log_level: {log_level}')
logger.info(f'Data connection string: {data_conn}')
logger.info(f'Data source directory: {data_conn}')
logger.info(f'Data processed directory: {data_conn}')

# Initialize database connection
conn = init_db(data_conn)

current_answer = ""
current_json = ""
prompt = ""

#Set up our pydantic models.
class Reference(BaseModel):
    folder: str
    document: str
    page: str
    paragraph: str
    content: str

class Content(BaseModel):
    summary: str
    create_date: date
    last_updated: date
    token_count: int
    references: List[Reference]

class Action(BaseModel):
    create_date: date
    due_date: date
    priority: str
    reference: str
    who: str
    name: str
    description: str
    next_steps: str

class ReviewDate(BaseModel):
    date: date
    instruction: str
    findings: str
    actions: List[Action]

class Contract(BaseModel):
    contact_name: str
    contract_number: str
    client_name: str
    date_created: date
    number_of_pages: int
    header: str
    footer: str
    review_dates: List[ReviewDate]
    content: List[Content]

contract_schema = {
    "name" : "contract_summary",
    "description" : "output object for a contract summary",
    "parameters" : Contract.schema(),
    "return_type" : "json" 
    } 
logger.debug(f'contract schema: {contract_schema}')

class ContentSection(BaseModel):
    page_number: int
    summary: str
    reference: List[str]

class Document(BaseModel):
    document_name: str
    document_directory: str
    document_processed_date: str
    content_sections: List[ContentSection]

content_schema = {
    "name" : "content_summary",
    "description" : "output object for a content summary",
    "parameters" : Content.schema(),
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
logger.debug(f'expert schema: {expert_schema}')

#Grab the pdf files from the source directory and process them into page sized bites.
def split_pdfs_in_directory():
    # Ensure the output directory exists
    if not os.path.exists(data_pre_processed):
        logger.debug(f'Creating pre processing directory: {data_pre_processed}')
        os.makedirs(data_pre_processed)
    else:
        logger.debug(f'Found pre processing directory: {data_pre_processed}')

    # Ensure the processed directory exists
    processed_dir = os.path.join(data_source, 'processed')
    if not os.path.exists(processed_dir):
        logger.debug(f'Creating processed directory: {processed_dir}')
        os.makedirs(processed_dir)
    else:
        logger.debug(f'Found processed directory: {processed_dir}')

    # List all files in the directory
    for filename in os.listdir(data_source):
        if filename.endswith('.pdf') and not os.path.isdir(os.path.join(data_source, filename)):
            file_path = os.path.join(data_source, filename)
            logger.debug(f'Processing file: {file_path}')
            
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                logger.debug(f'Found {total_pages} pages in {filename}')

                # Create the JSON data object
                document_data = {
                    "document_name": filename,
                    "document_directory": data_source,
                    "document_processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pages": []
                }

                # Extract content for each page and add to the JSON data object
                for page_num in range(total_pages):
                    page_content = pdf_reader.pages[page_num].extract_text()
                    token_count =len(encoding.encode(page_content))
                    document_data["pages"].append({
                        "page_number": page_num + 1,
                        "token_count": token_count,
                        "content": page_content
                    })
                logger.debug(f'Processed {total_pages} pages in {filename}')

                # Save the JSON data object to the reference directory
                output_filename = f"{filename[:-4]}.json"
                output_path = os.path.join(data_pre_processed, output_filename)
                with open(output_path, 'w') as json_file:
                    json.dump(document_data, json_file, indent=4)

                logger.info(f"Processed {filename} and saved JSON to {output_path}")

            # Move the processed file to the processed directory
            move_to_processed(file_path, processed_dir)
            logger.info(f"Moved {filename} to processed directory")

    logger.info("PDF processing completed.")

def move_to_processed(file_path, processed_dir):
    shutil.move(file_path, processed_dir)

def process_json(token_limit):
    # List all .json files in the directory
    json_files = [f for f in os.listdir(data_pre_processed) if f.endswith('.json')]
    logger.debug(f'Found {len(json_files)} JSON files in {data_pre_processed}')
    
    if not json_files:
        return {"document_name":"No files found for pre processing"}

    # Find the youngest .json file
    youngest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(data_pre_processed, f)))
    file_path = os.path.join(data_pre_processed, youngest_file)

    with open(file_path, 'r') as f:
        data = json.load(f)

    total_tokens = 0
    processed_sections = []

    for page in data["pages"]:
        # Skip sections that have processed=True
        if page.get('processed') == True:
            logger.debug(f'Skipping page {page["page_number"]} as it has already been processed')
            continue
        else:
            logger.debug(f'Processing page {page["page_number"]}')

        section_tokens = page["token_count"]
        logger.debug(f'''
                     -----------------------------------
                        Processing Page: 
                     ------------------------------
                     {page}''')

        if total_tokens + section_tokens <= token_limit:
            total_tokens += section_tokens
            logger.debug(f'''Total tokens: {total_tokens}
                     ---------------------------''')
                         
            page['processed'] = True
            processed_sections.append(page)
        else:
            logger.debug(f'''
                     ----------------------------------
                        Finished run at page {page["page_number"]}
                        Total tokens: {total_tokens}
                        Token limit: {token_limit}
                     ----------------------------------''')
            break

    # Update the JSON object with processed sections
    new_json_object = {
        "document_name": data["document_name"],
        "document_directory": data["document_directory"],
        "document_processed_date": data["document_processed_date"],
        "pages": processed_sections
    }
    logger.debug(f'JSON object: {new_json_object}')

    # Save the updated JSON object (overwriting the existing file)
    for page in processed_sections:
        data["pages"][page["page_number"] - 1] = page  # Assuming page_number starts from 1

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    # Check if all sections have been processed
    all_processed = all(page.get('processed') == True for page in data["pages"])

    # Move the processed file to the ../processed directory if all sections have been processed
    if all_processed:
        logger.debug(f'Moving {file_path} to {data_processed}')
        shutil.move(file_path, os.path.join(data_processed, youngest_file))

    return new_json_object

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
    prompttext = '''role:
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
      ],
      functions = [contract_schema],
      function_call = {"name":"contract_summary"}
    )
    logger.debug(response)

    if response['choices'] and response['choices'][0]['message']: #type: ignore
      logger.info(response['choices'][0]['message']['content'].strip()) #type: ignore
      return response['choices'][0]['message']['content'].strip() #type: ignore

def get_content():
    logger.debug(f'Prompt: {prompt}')
    logger.debug(f'content schema: {content_schema}')
    logger.debug(f'current_json: {current_json}')

    prompttext = f'''{prompt} Dont populate the reviewers list at this stage.
    INPUT:
    {current_json}
    '''

    logger.debug(f'''----------------------
                 current_prompt: 
                 -------------------------
                 {prompttext}
                 -------------------------''')

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
        {
            "role": "assistant",
            "content": prompttext
        }
      ],
      functions = [content_schema],
      function_call = {"name":"content_summary"}
    )
    logger.debug(f'response:{response}')

    #if we have a response
    if response.choices[0] and response.choices[0]['message']:
      return_json = json.loads(response.choices[0]["message"]["function_call"]["arguments"]) #type: ignore
      logger.info(f'First Answer run: {return_json}')
      logger.debug(f'''First Answer message: {response.choices[0]['message']}''')
      return return_json
    else:
        logger.error(f'Chatgpt output: {response}')
        return {"Error": "Chat GPT Error"}

def get_experts():
    prompttext = '''I want a response to the following question:
    
    ''' + prompt + '''

        Name 3 world-class experts (past or present) who would be great at answering this?
        Don't answer the question yet. Just name the experts.
        '''
    logger.info(prompttext)

    response = openai.ChatCompletion.create(
        model="gpt-4",
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
    #grab the source files, split them into pages, and stick them into a preprocessed directory.
    split_pdfs_in_directory()

    #build the input dataset, get together a total of 1k tokens.
    current_json = process_json(2000)

    prompt = '''
    Role:
    You are a Program Directory in a large IT outsourcing company.

    Background:
    Your task involves reviewing contracts and pulling from them all of the pertinent information that will impact the delivery of your projects.
    For refernence use the prince2 project methodology as a guide for understanding what elements are important to a project.

    Task:
    Process the text below sentence by sentence, review each sentence and determine if it has changed subject from the previous sentence.
    Remeber that the subject of a sentence is whats relevent to managing a project.
    Every time a sentence changes subject, create a new content section in the output json object, add a sumary of the content, then and add all the sentences since the last subject change as reference objects, splitting them as appropriate if the subject crosses paragraphs.
    If json object already has a content section, take care to track the relevent page change.
    If the sentence is a header or footer, and the header or footer section of the json object is blank, then add it, otherwise ignore it.'''


    #Get the first answer
    current_answer = get_content()
    logger.info(current_answer)

    #get some experts to delibarate
    #experts = get_experts()
    


    # Now we go and get the reviewers comments
    #data = json.loads(experts)
    #for expert in data['experts']:
    #    current_answer = get_review(expert['Name'])
    #    logger.info(current_answer)

    #current_answer = get_final_answer()
    #logger.info(current_answer)