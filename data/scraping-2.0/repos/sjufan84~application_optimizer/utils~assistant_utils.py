""" Helper functions to generate the correct assistant instructions """
# Initial imports
import sys
from typing import List, Optional
import os
from pydantic import BaseModel, Field
from openai import OpenAI
import streamlit as st
from fastapi import UploadFile
from dotenv import load_dotenv
# Add the path of the package to the system path
sys.path.append('C:/Users/sjufa/OneDrive/Desktop/Current Projects/pr_prophet/backend')

# Load environment variables
load_dotenv()

# Set OpenAI API key
api_key = os.getenv("OPENAI_KEY2")
organization = os.getenv("OPENAI_ORG2")

# Set up the client
client = OpenAI(api_key=api_key, organization=organization, max_retries=3, timeout=10)

if "file_ids" not in st.session_state:  
    st.session_state.file_ids = []

core_models = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]

class OptimizeResponse(BaseModel):
    """ The response from the optimization """
    keywords: List[str] = Field(..., description="The keywords extracted from the application text")
    suggestions: List[str] = Field(..., description="The suggestions for the application text")
    updated_text: Optional[str] = Field(..., description="The updated application text including the suggestions")

# Define a function to receive and upload files to the assistant
def upload_files(files: List[UploadFile]):
    """ Takes in the list of file objects and uploads them to the assistant """
    file_ids_list = []
    for file in files:
        # Upload the file
        response = client.files.create(file=file, purpose="assistants")
        # Append the file url to the list
        file_id = response.id
        file_ids_list.append(file_id)

    st.session_state.file_ids = st.session_state.file_ids + file_ids_list

    return file_ids_list

def create_assistant_files(file_ids: List[str]):
  """ Create assistant files from the file ids """
  assistant_files_list = []
  for file in file_ids:
    if file not in st.session_state["assistant_files"]:
      assistant_file = client.beta.assistants.files.create(
      assistant_id="asst_FNbc0H90UCKiCu7DR41Ot3Mp", 
      file_id=file
    )
    assistant_files_list.append(assistant_file)

  return assistant_files_list 

# Create a function to extract keywords and suggestions from the assistant
def process_application(job_description: str, application_text : str):
    """ Process the application text and return the keywords and suggestions """
    # Create the run
    messages = [
      {
        "role" : "system", 
        "content" : f"You are helpful HR employee helping an applicant and\
        the business better optimize or create an applicants resume to better\
        reflect their fit for the job.  Your response should be a JSON object\
        with the same fields as the {OptimizeResponse} object.  These fields\
        are keywords, suggestions, and updated_text."
    }
    ]
    for model in core_models:
      try:
        # Create the API call
        response = client.chat.completions.create(
          messages = messages,
          model = model,
          top_p = 1,
          temperature = 0.75,
          max_tokens = 750,
          response_format = {"type" : "json_object"}
          )
        # Get the chef response
        process_response = response.choices[0].message.content
        return process_response
      except TimeoutError as e:
        print(e)
        continue

# Create a function to list the files associated with the assistant
def list_assistant_files(assistant_id: str = "asst_FNbc0H90UCKiCu7DR41Ot3Mp"):
  """ List the files associated with the assistant """
  assistant_files_list = []
  assistant_files = client.beta.assistants.files.list(
    assistant_id=assistant_id
  )
  for data in assistant_files.data:
    assistant_files_list.append(data.id)
  return assistant_files_list

def delete_assistant_files(assistant_id: str = "asst_FNbc0H90UCKiCu7DR41Ot3Mp", file_ids: List[str] = None):
  """ Delete the files associated with the assistant """
  deleted_files = []
  for file in file_ids:
    deleted_assistant_file = client.beta.assistants.files.delete(
      assistant_id=assistant_id,
      file_id=file
    )
    deleted_files.append(deleted_assistant_file)

  return deleted_files
