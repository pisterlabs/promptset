import os.path
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io
from openai import OpenAI
import re
import math
import json
import requests
import weaviate
from weaviate.exceptions import UnexpectedStatusCodeException
import PyPDF2


def main():
  
  service = get_google_drive_api_service()

  items = get_drive_metadata_list(service)

  text_blocks = create_text_blocks(service, items)

  create_weaviate_data(text_blocks)


#Define parameters and variables needed for Google Drive API  connection and return Google Drive API service.
def get_google_drive_api_service():

  # Google API scopes: If modifying these scopes, delete the file token.json.
  SCOPES = ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/drive.metadata.readonly"]

  #Drive v3 API. Prints the names and ids of the first n files the user has access to.
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  return build("drive", "v3", credentials=creds)


# Call the Google drive API to retrieve a list of filenames (and other metadata).
# List returned only includes metadata, not the content of the file.
def get_drive_metadata_list(service):

    query_str = "not ('aino.nyblom@qbizinc.com' in owners)" \
              + "and not (name contains 'career development' or name contains 'Career Development') " \
              + "and (name contains '.pdf' or mimeType='application/vnd.google-apps.document'" \
              + "or mimeType='application/vnd.google-apps.presentation' or mimeType='application/vnd.google-apps.spreadsheet')"

    results = (
        service.files()
        .list(q=query_str, pageSize=1000, includeItemsFromAllDrives=True, supportsAllDrives=True, fields="nextPageToken, files(id, name, mimeType)") #max page size = 1000
        .execute()
    )

    items = results.get("files", [])
    print(len(items), " files found")

    if not items:
      print("No files found.")
      return None

    else:
      return items
      

# Iterate through the list of Google drive file metadata, and call functions to read each file content and create text blocks for the files.
# Google document and PDF files need different handling.
def create_text_blocks(service, items):

    text_blocks = []
    
    print("Files:")
    
    for item in items:
      print(f"{item['name']} ({item['id']}) ({item['mimeType']})")
      text_blocks.extend(text_blocks_for_google_doc_and_pres(item, service))
      text_blocks.extend(text_blocks_for_pdfs(item, service))                     

    return text_blocks


# Extract the text content of a google document or presentation file, and call a function to create 300 word text blocks.
# Reference: https://medium.com/@matheodaly.md/using-google-drive-api-with-python-and-a-service-account-d6ae1f6456c2
def text_blocks_for_google_doc_and_pres(item, service):
  
  if item['mimeType'] in ("application/vnd.google-apps.document", "application/vnd.google-apps.presentation"):
    try:
      request_file = service.files().export_media(fileId=item['id'], mimeType='text/plain').execute()
      text = str(request_file)
    except HttpError as error:
      print(f"An error occurred: {error}")
      return []
    return text_blocks_for_a_file(item['name'], text, 300, 10, True)
  else:
    return []

# Extract the text content of a google spreadsheet file, and call a function to create 300 word text blocks.  
def text_blocks_for_google_spreadsheet(item, service):
  
  if item['mimeType'] == "application/vnd.google-apps.spreadsheet":
    try:
      request_file = service.files().export_media(fileId=item['id'], mimeType='text/csv').execute()
      text = str(request_file)
    except HttpError as error:
      print(f"An error occurred: {error}")
      return []
    return text_blocks_for_a_file(item['name'], text, 300, 10, True)
  else:
    return []

  
# Extract the text content of a pdf file, and call a function to create 300 word text blocks.
def text_blocks_for_pdfs(item, service):
  if item['name'][-3:] == "pdf":
    request_file = service.files().get_media(fileId=item['id'])
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request_file)
    done = False
    while done is False:
      status, done = downloader.next_chunk()
      file_retrieved: str = file.getvalue()
      file_io = io.BytesIO(file_retrieved)
      pdf_file = PyPDF2.PdfReader(file_io)
      text = ""

      # Loop through each page and extract text
      for page_num in range(len(pdf_file.pages)):
        page = pdf_file.pages[page_num]
        text += page.extract_text()
        return text_blocks_for_a_file(item['name'], text, 300, 10, True)
  else:
    return []

# Split text into blocks including n words. Return blocks in an array of dict (json) objects.      
def text_blocks_for_a_file(file_name, text_content, words_per_block, buffer_length, remove_urls):

    if remove_urls:
        text_content = re.sub(r'http\S+', '', text_content)
    list_of_words = text_content.split()
    # create word blocks
    n_o_blocks = math.ceil(len(list_of_words)/words_per_block)
    array_of_blocks = []
    for i in range(n_o_blocks):
        start_word = i*words_per_block
        end_word = (i+1)*words_per_block + buffer_length #take extra words (These will be overlapping between blocks, which is intended. Imperfect effort not to cut blocks in the middle of sentence) 
        block = {
            "block_id": f"{file_name}_block_{i}",
            "text_block": 'Using document ' + file_name + ' as a source:' + ' '.join(list_of_words[start_word:end_word])
        }
        array_of_blocks.append(block)
    return array_of_blocks

# Connect to Weaviate vector database using your endpoint (url), Weaviate API key, and openAI API key.
# Create Weaviate class and add the (already created) text blocks to the class.
def create_weaviate_data(text_blocks):
  
    client = weaviate.Client(
      url = "https://weviate-cluster-xom22sd6.weaviate.network",
      auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEVIATE_API_KEY"]),
      additional_headers = {
          "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]  
      }
    )

    class_obj = {
        "class": "Text_block",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {},
            "generative-openai": {}  # generative queries
        }
    }
    
    client.schema.delete_class("Text_block")
    
    try:
      client.schema.create_class(class_obj)
    except UnexpectedStatusCodeException as exception:
      print(exception)
      pass

    client.batch.configure(batch_size=100)  # Configure batch

    with client.batch as batch:  # Initialize a batch process
        for i, d in enumerate(text_blocks):  # Batch import data
            print(f"importing text-block: {i+1}", d["block_id"])
            properties = {
                "text_block": d["block_id"],
                "text_block": d["text_block"],
            }
            batch.add_data_object(
                data_object=properties,
                class_name="Text_block"
            )

            
if __name__ == "__main__":
  main()
