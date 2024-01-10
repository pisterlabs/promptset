import time
import openai
import pandas as pd
import json
import random
import os
from tqdm import tqdm_notebook as tqdm
import pathlib as pl
import shutil

from urllib3 import HTTPSConnectionPool

OPENAPI_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAPI_KEY

input_data = pl.Path("/home/murilo/RelNetCare/data/processed/dialog-re-12cls-with-no-relation-undersampled-llama/dialog-re-12cls-with-no-relation-undersampled-llama-test.json")
data_stem = input_data.parents[0].name
output_dir = pl.Path('/home/murilo/RelNetCare/data/processed/chat-gpt-predictions') / data_stem

os.makedirs(output_dir, exist_ok=True)

with open(input_data, encoding='utf8') as fp:
    json_data = json.load(fp)

def safe_openai_call(prompt, attempts=3, timeout_duration=30):
    for attempt in range(attempts):
        try:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                timeout=timeout_duration,  # Set your timeout duration here
                request_timeout =timeout_duration  # Set your timeout duration here
            )
        except HTTPSConnectionPool as e:
            if attempt < attempts - 1:
                print(f"API call failed... Waiting to retry again!")
                # Wait before retrying, could implement exponential backoff here
                time.sleep(2 ** attempt)
            else:
                print(f"API call failed after {attempts} attempts: {e}")
                return None  # or handle it in a way that fits your flow


# Initialize a DataFrame

processed_ids = {f.split('.json')[0] for f in os.listdir(output_dir) if f.endswith('.json')}

df = pd.DataFrame(columns=['human_id', 'human_prompt', 'ground_truth_response', 'open_ai_predicted_response'])

# Filter json_data to exclude already processed items
filtered_json_data = [entry for entry in json_data if entry['id'] not in processed_ids]

pbar = tqdm(total=len(filtered_json_data))
last_predictions = []
data = []
# Loop through each entry in your JSON data
for entry in filtered_json_data:
    human_id = entry['id']
    processed_ids = [f.split('.json')[0] for f in os.listdir(output_dir) if f.endswith('.json')]
    for convo in entry['conversations']:
        if convo['from'] == 'human':
            human_prompt = convo['value']
            # This is your ground truth if available, or None
            ground_truth_response = next((c['value'] for c in entry['conversations'] if c['from'] == 'gpt'), None)
            
            try: 
                # Send the prompt to OpenAI's ChatGPT (replace "your_model_name" with the actual model name)
                response = safe_openai_call(human_prompt)
                
                # Get OpenAI's predicted response
                open_ai_predicted_response = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                exception = None
            except Exception as e:
                exception = str(e)
                open_ai_predicted_response = None
               
            # Update last predictions list
            last_predictions.append(open_ai_predicted_response)
            # Keep only the last five predictions
            last_predictions = last_predictions[-3:]
            
            # Update the tqdm progress bar description
            pbar.set_description(f"Last 3 Predictions: {last_predictions}")
            pbar.update(1)  # Update the progress

            data_row = {
                'human_id': human_id,
                'human_prompt': human_prompt,
                'ground_truth_response': ground_truth_response,
                'open_ai_predicted_response': open_ai_predicted_response,
                'exception': exception
            }
            
            with open(output_dir / f"{human_id}.json", mode='w') as fp:
                fp.write(json.dumps(data_row, indent=2))

            # Append to the DataFrame
            data.append(data_row)
            
            

os.makedirs(output_dir / 'timeout', exist_ok=True)

for f in os.listdir(output_dir):
    if f.endswith('.json'):
        with open(output_dir / f, encoding='utf8') as fp:
            data = json.load(fp)

        if data['exception']:
            shutil.move(output_dir / f, output_dir / 'timeout' / f)
            print('moved!')
