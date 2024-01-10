import os
import openai
import json
from datetime import datetime
import calendar


def write_log(prompt, ai_response, unique_id):
    log_folder = 'gpt_logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    file_path = os.path.join(log_folder, unique_id + '.txt')
    
    with open(file_path, 'a') as file:
        file.write(prompt + '\n')
        file.write(ai_response + '\n')

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def load_conversations_by_id(results):
    tmp_results = []
    if 'matches' in results:
        for n in results['matches']:
            info = load_json('local_database/%s.json' % n['id'])
            tmp_results.append(info)
        ordered = sorted(tmp_results, key=lambda d: d['time'], reverse=False)
        messages = [i['message'] for i in ordered]
        return '\n'.join(messages).strip()
    else:
        return "No 'matches' key found in the results dictionary."

def save_metadata_to_json(metadata, unique_id):
    folder_path = "local_database"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Generate the file path
    file_path = os.path.join(folder_path, f"{unique_id}.json")
    
    # Save metadata to the JSON file
    with open(file_path, "w", encoding='utf-8') as outfile:
        json.dump(metadata, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def gpt3_embeddings(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode() # fix all unicode errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']
    return vector

def timestamp_to_string(timestamp):
    # Convert timestamp to datetime object
    dt_object = datetime.fromtimestamp(timestamp)
    
    # Extract year, month, day, hour, minute, and second from datetime object
    year = dt_object.year
    month = calendar.month_name[dt_object.month]
    day = dt_object.day
    hour = dt_object.hour
    minute = dt_object.minute
    second = dt_object.second
    
    # Format datetime components as readable string
    formatted_string = f"{month} {day}, {year} {hour:02d}:{minute:02d}:{second:02d}"
    
    return formatted_string