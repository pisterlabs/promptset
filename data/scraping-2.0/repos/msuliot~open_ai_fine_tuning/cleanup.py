import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def delete_file(file_id):
    try:
        openai.File.delete(file_id)
        print("File deleted successfully")
    except Exception as e:
        print("Error deleting file: ", e)


def delete_finetune_model(model_id):
    try:
        openai.Model.delete(model_id)
        print("Model has been deleted successfully")
    except Exception as e:
        print("Error deleting model: ", e)


def download_file(file_id, filename="downloaded.jsonl"):
    try:
        # Download the file
        the_file = openai.File.download(file_id)
        data_str = the_file.decode('utf-8')
        with open(filename, 'w') as file:
            file.write(data_str)
        print("File downloaded successfully")
    except Exception as e:
        print("Error downloading file: ", e)

def delete_all_files():
    file_list = openai.File.list()
    for file in file_list['data']:
        print(file['id'], file['purpose'], file['status'])
        delete_file(file['id'])

def delete_all_models():
    model_list = openai.FineTuningJob.list(limit=50)
    for model in model_list['data']:
        print(model['status'], model['fine_tuned_model'])
        delete_finetune_model(model['fine_tuned_model'])

def list_files():
    print("\n===== File List =====")
    file_list = openai.File.list()
    for file in file_list['data']:
        print(file['id'], file['purpose'], file['status'])

def list_models():
    print("\n===== Model List =====")
    model_list = openai.FineTuningJob.list(limit=50)
    for model in model_list['data']:
        print(model['status'], model['fine_tuned_model'])


# delete_file("file-5tZ09GT4pGuTAYuBmRHjYgGO")  
# delete_all_files()
#  
# delete_finetune_model("ft:gpt-3.5-turbo-0613:michael-ai::7rugXpfD")
# delete_all_models()
#
# download_file("file-O4IZuDzXVaPvE5XUPdZpwKJg","down.jsonl")

list_files()
list_models()