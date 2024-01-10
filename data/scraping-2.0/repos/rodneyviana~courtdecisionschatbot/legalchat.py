#Note: This code sample requires OpenAI Python library version 0.28.1 or lower. See requirements.txt for more information.
import os
import openai
import dotenv
import json
import time
import logging
from document_analysis import JudgeDecision2116278PDF

from PyPDF2 import PdfReader

def pdf_to_text(file_path):
  with open(file_path, 'rb') as file:
    reader = PdfReader(file)
    text = ''
    for page_number in range(len(reader.pages)):
      page = reader.pages[page_number]
      text += f"\n{page.extract_text()}"
  return text

dotenv.load_dotenv()

log_file = "legalchat.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')

logging.info("Starting legalchat.py")
pdfFilePath = "judge-decision-21-16278.pdf"
fullText = '';  
use_advanced: str = input("Use advanced text extraction? (Y/n):");

if(not use_advanced.lower().startswith('n') ):
  logging.info("Using advanced text extraction")
  docana: JudgeDecision2116278PDF = None
  with open("judge-decision.json", "r", encoding='utf-8') as read_file:
      print("Reading Analysis file")
      temp = json.load(read_file) #, object_hook=JudgeDecision2116278PDF);
      docana = JudgeDecision2116278PDF(temp)
  for paragraph in docana.analyzeResult['paragraphs']:
    fullText = f"{fullText}\n{paragraph['content']}"
else:
  logging.info("Using basic text extraction")
  fullText = pdf_to_text(pdfFilePath)

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
   
conversation = [
                {
                   "role":"system",
                   "content": f"You are a lawyer and you will respond questions about this legal document: {fullText}"
                }
               ]
keepGoing = True


while keepGoing:
  try:
    nextMessage = input("what is your question? (enter exit):")
    if(len(nextMessage) > 0):
      logging.info(f"User question: {nextMessage}")
      conversation.append({
          "role": "user",
          "content": nextMessage
        })
      logging.info("Sending question to OpenAI")
      completion = openai.ChatCompletion.create(
        engine=os.getenv("OPENAI_ENGINE_ID"),
        messages = conversation,
        temperature=0.5,
        max_tokens=4000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
      )
      conversation.append({
           "role": completion.choices[0].message.role,
           "content": completion.choices[0].message.content
           })
      logging.info("Response received from OpenAI")
      logging.info(f"Response:\n{completion.choices[0].message.content}")
      print(f"\nResponse:\n{'=' * 80}")
      print(completion.choices[0].message.content)
      print('=' * 80)
    keepGoing = len(nextMessage) > 0
  #except RateLimitError as error:
  #  print(error)
  except Exception as error:
    print(error)
    logging.error(error)
    keepGoing = False
    if(error.http_status == 429):
      if("Retry-After" in error.headers):
        sleep_time = int(error.headers["Retry-After"])
        logging.info(f"Rate limit exceeded. Waiting {sleep_time}s for retry-after header...")
        print(f"Rate limit exceeded. Waiting {sleep_time}s for retry-after header...")
        time.sleep(sleep_time)
        keepGoing = True
      else:
        print(error.user_message);
        logging.error(error.user_message)
        quit = input("Please wait at least one minute and press enter to continue 'q' to quit...")
        if(quit.lower().startswith('q')):
          keepGoing = False
        else:
          keepGoing = True
    else:
      print(f"Error: {error}");
      print("Unexpected error. Exiting...")
      logging.error(f"Unexpected error: {error}")
        
print("good bye!")
logging.info("Exiting legalchat.py")

