from dotenv import load_dotenv
import openai
import os
import pygraphviz as pgv

load_dotenv()
openai.api_key=os.getenv('OPENAI_API_KEY')

def upload():
    response=openai.File.create(
        file=open('../dsa/refined_dsa_english.jsonl','rb'),
        purpose='fine-tune')
    print(response) 

def fine_tune():
    file_id='ABC'
    response=openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-3.5-turbo-1106")
    print(response)

def get_file_status():
    try:
        file_id='ABC'
        file_info = openai.File.retrieve(file_id)
        print(file_info.status)
        print(file_info.status_details)
    except openai.error.OpenAIError as e:
        print("Error while retrieving file status:", e)
        return None
    
user_input = input(
  "\nScegli:"+
  "\n1. upload jsonl file"+
  "\n2. fine-tune"+
  "\n3. status"+
  "\n\n>")

if (user_input == "1"):
    upload()
elif (user_input == "2"):
    fine_tune()
elif (user_input == "3"):
    get_file_status()
else:
    print("INVALID SELECTION")