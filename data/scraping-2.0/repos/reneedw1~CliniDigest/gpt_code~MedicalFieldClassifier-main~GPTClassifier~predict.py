import openai
import regex as re
import time
from process_data import *

# TODO: fill in openai.api_key
openai.api_key = ''
max_retries = 5
retry_delay = 5  # seconds

def get_completion(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def predict(csv_file_path):
    clinical_trials = read_csv_to_dict(csv_file_path)
    clinical_trials_descriptions = clinical_trials["Description"]

    full_results = []
    medical_fields = ["Somnology", "Gynecology", "Obstetrics", "Cardiology", "General Physiology", "Endocrinology", "Bariatrics", "Psychiatry", "Oncology", "Gastroenterology", "Pulmonology", "Chronic pain or diseases", "Nephrology", "Other"]
    for clinical_trial_description in clinical_trials_descriptions:
        prompt = f'''You are provided with a set of 14 medical field classes: \"Somnology\", \"Gynecology\", \"Obstetrics\", \"Cardiology\", \"General Physiology\", \"Endocrinology\", \"Bariatrics\", \"Psychiatry\", \"Oncology\", \"Gastroenterology\", \"Pulmonology\", \"Chronic pain or diseases\", \"Neprhology\", and \"Other\". 
Your task is to analyze the given clinical trial description and classify it into one of these 14 medical field classes. Please provide only one field as your output.
---
Trial Description: {clinical_trial_description[0:3000]}
---
Task: Classify the clinical trial into one of the 14 specified medical field classes:
- Somnology
- Gynecology
- Obstetrics
- Cardiology
- General Physiology
- Endocrinology
- Bariatrics
- Psychiatry
- Oncology
- Gastroenterology
- Pulmonology
- Chronic pain or diseases
- Nephrology
- Other
Please provide only the medical field name as your output.
'''        
        
        response = ''
        for retry in range(max_retries):
            try:
                response = get_completion([{"role": "user", "content": prompt}])
                break  

            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        else:
            print("Max retries reached. Could not complete the action.")

        if response == 'Hematology':
            response = 'Oncology'
        for field in medical_fields:
            if field in response:
                response = field
                break
        else:
            response = 'Other'
        full_results.append(response)
    name = re.match('.*\/(.*)TrialsToAdd.csv', csv_file_path).group(1)
    with open(f'{dir}gpt_code/MedicalFieldClassifier-main/GPTClassifier/predictions_{name.lower()}_gpt.txt', 'w') as output:
        output.writelines('\n'.join(full_results))
