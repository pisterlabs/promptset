import json
import os
import glob
import pandas as pd
import logging
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import openai

# Initialize logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    logging.error("No OpenAI API Key found.")
    exit(1)

# Helper functions
def read_file_content(file_name):
    try:
        with open(file_name, 'r') as f:
            return f.readlines()
    except Exception as e:
        logging.error(f"Failed to read {file_name}: {e}")
        return []

def save_to_json(data, filename):
    try:
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except Exception as e:
        logging.error(f"Failed to save to {filename}: {e}")

def collate_fine_tune_data():
    dialogues = []
    system_message = {
        "role": "system",
        "content": "Your task is to accurately represent professional and educational background, as well as interests and hobbies, while speaking in the first person. Kindly refrain from answering questions that aren't related to these topics."
    }
    
    for file_name in glob.glob('./src/text_files/*questions.txt'):
        lines = read_file_content(file_name)
        
        question, answer = None, None
        for line in lines:
            line = line.strip()
            if line.startswith("Question:"):
                question = line[len("Question:"):].strip()
            elif line.startswith("Answer:"):
                answer = line[len("Answer:"):].strip()
            
            if question and answer:
                dialogue = {"messages": [system_message, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]}
                dialogues.append(dialogue)
                question, answer = None, None
    
    save_to_json(dialogues, './src/collated_questions.json')
    logging.info("Successfully collated data for fine-tuning")

def split_data():
    try:
        with open("./src/collated_questions.json", "r") as f:
            data = json.load(f)
        
        df = pd.json_normalize(data)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_json("./src/train_data.jsonl", orient="records", lines=True)
        val_df.to_json("./src/val_data.jsonl", orient="records", lines=True)
        logging.info("Successfully created train/test split")
    except Exception as e:
        logging.error(f"Failed to split data: {e}")

def upload_train_test_files():
    try:
        training_response = openai.File.create(file=open('./src/train_data.jsonl', "rb"), purpose="fine-tune")
        training_file_id = training_response["id"]
        validation_response = openai.File.create(file=open('./src/val_data.jsonl', "rb"), purpose="fine-tune")
        validation_file_id = validation_response["id"]
        
        logging.info(f"Training file ID: {training_file_id}\nValidation file ID: {validation_file_id}")
        logging.info("Allow 30 seconds for files to upload")
        
        return training_file_id, validation_file_id
    except Exception as e:
        logging.error(f"Failed to upload train/test files: {e}")

def create_finetune_job(training_file_id, validation_file_id, suffix):
    try:
        response = openai.FineTuningJob.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model="gpt-3.5-turbo",
            suffix=suffix,
        )
        job_id = response["id"]
        logging.info(f"Job ID: {response['id']}\nStatus: {response['status']}")
        
        return job_id
    except Exception as e:
        logging.error(f"Failed to create fine-tuning job: {e}")

def check_job(job_id):
    try:
        response = openai.FineTuningJob.list_events(id=job_id, limit=100)
        for event in reversed(response["data"]):
            logging.info(event["message"])
    except Exception as e:
        logging.error(f"Failed to check job: {e}")

if __name__ == "__main__":
    pass
