import openai
import logging
from config import load_config

openai.api_key = load_config()["OPENAI"]["API_KEY"]

import json

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_db_config(db_type):
    try:
        with open(f"{db_type.lower()}_config.json", 'r', encoding='utf-8') as f:
            db_config = json.load(f)
        logging.info(f'Successfully loaded db config for {db_type}: {db_config}')
        return db_config
    except Exception as e:
        logging.exception(e)
        raise

def create_ai_response(db_type, text, previous_qusetion=None, previous_answer=None):
    try:
        config = load_db_config(db_type)
        messages = []

        system_message = '\n'.join(config['system_message'])
        messages.append({"role": "system", "content": system_message})

        for sample in config['samples']:
            input = sample['input']
            output = '\n'.join(sample['output'])
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})

        if previous_answer and previous_qusetion:
            messages.append({"role": "user", "content": previous_qusetion})
            messages.append({"role": "assistant", "content": previous_answer})

        messages.append({"role": "user", "content": text})

        master_ai_response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages,
            max_tokens=6000,
            n=1,
            stop=None,
            temperature=0.5,
        ).choices[0]['message'].content.strip()

        logging.info(f'Successfully created AI response: {master_ai_response}')

        return master_ai_response
    except Exception as e:
        logging.exception(e)
        raise

def transform_input_to_sql(db_type, user_input): 
    try:
        result = create_ai_response(db_type, user_input)

        if "UNRESOLVABLE_QUERY" in result:
            raise Exception("{0}".format(result.split('UNRESOLVABLE_QUERY')[1]))
        
        logging.info(f'Successfully transformed input to SQL: {result}')
        return result
    except Exception as e:
        logging.exception(e)
        raise

def retry_transform_input_to_sql(db_type, user_input, ai_answe, db_error):
    try:
        result = create_ai_response(db_type, db_error, user_input, ai_answe)

        if "UNRESOLVABLE_QUERY" in result:
            raise Exception("{0}".format(result.split('UNRESOLVABLE_QUERY')[1]))
        
        logging.info(f'Successfully retried and transformed input to SQL: {result}')
        return result   
    except Exception as e:
        logging.exception(e)
        raise
