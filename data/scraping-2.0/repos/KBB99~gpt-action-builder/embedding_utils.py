import openai
import os
import pandas as pd
from config import CSV_FILE_PATH, CSV_COLUMNS, ACTIONS_CSV_FILE_PATH, ACTIONS_CSV_COLUMNS

def generate_embedding(input):
    return openai.Embedding.create(input=input, model="text-embedding-ada-002")['data'][0]['embedding']
    
def load_or_create_dataframe():
    if os.path.isfile(CSV_FILE_PATH):
        return pd.read_csv(CSV_FILE_PATH)
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)
    
def load_or_create_actions_dataframe():
    if os.path.isfile(ACTIONS_CSV_FILE_PATH):
        return pd.read_csv(ACTIONS_CSV_FILE_PATH)
    else:
        return pd.DataFrame(columns=ACTIONS_CSV_COLUMNS)
    
def update_dataframe_based_on_evaluation(df, goal, david_instantiation_prompt, constraints, tips, execution_output, evaluation_output, extracted_actions_output):
    print("Extracted actions:", extracted_actions_output)
    actions_embedding = generate_embedding(input=extracted_actions_output)
    status = 'succeeded' if 'yes' in evaluation_output.strip().lower() else 'false'
    new_row = {
        'Goal': goal, 
        'InstantiationPrompt': david_instantiation_prompt, 
        'Constraints': constraints, 
        'Tips': tips, 
        'Actions': extracted_actions_output, 
        'embedding': actions_embedding, 
        'execution': execution_output, 
        'status': status
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE_PATH, index=False)
    return extracted_actions_output

def update_actions_dataframe(df, extracted_actions_output):
    print("Extracted actions:", extracted_actions_output)
    actions = extracted_actions_output.split('|||')
    action_entries = []

    for action in actions:
        embedding = generate_embedding(action)
        action_entries.append({
            'action': action,
            # 'action_description': description,
            'action_embedding': embedding
        })

    # Append the new action entries to the original dataframe
    df = pd.concat([df, pd.DataFrame(action_entries)], ignore_index=True)
    df.to_csv(ACTIONS_CSV_FILE_PATH, index=False)

    return extracted_actions_output