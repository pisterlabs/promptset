import openai
import os
from scripts import utils
from dotenv import load_dotenv

def get_baseline_response(
    question:str, 
    db_id:str, 
    top_matches: list, 
    schemas:dict):
    """
    Function to get baseline sql from LLM
    """
    # Setup OpenAI API
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    model = utils.get_model()
    
    # Build prompt
    messages = []
    
    # Build system instruction
    db_type = 'sqlite'
    target_schema = utils.generate_schema_code(db_id, schemas)
    system_instruction = {
        "role": "system",
        "content": f'Given a schema and a question in English, generate read only sql code for {db_type}. Respond with sql code only and do not include any explanation of code in response. If you are unsure, respond with "Error".\n\nImportant: Always use joins or set operations over nested queries. Never use left joins in queries.\n\n' + target_schema
    }
    messages.append(system_instruction)
    
    # Build few shot examples
    for top_match in top_matches:
        few_shot_user = {
            "role": "user",
            "content": top_match['question']
        }
        messages.append(few_shot_user)
        few_shot_assistant = {
            "role": "assistant",
            "content": top_match['query']
        }
        messages.append(few_shot_assistant)
    
    # Build user question
    user_question = {
        "role": "user",
        "content": question
    }   
    messages.append(user_question)

    # Get response from OpenAI API
    try:
        response = utils.call_openai(
            model=model['model_name'],
            messages=messages,
            temperature=model['temperature'],
            max_tokens=model['max_tokens'],
            top_p=model['top_p'],
            frequency_penalty=model['frequency_penalty'],
            presence_penalty=model['presence_penalty']
        )
    except:
        print('[ERROR] OpenAI API call failed for baseline response.')
        return None
    
    # Return response
    return {
        'sql': response['choices'][0]['message']['content'],
        'target_schema': target_schema,
        'input_tokens': response['usage']['prompt_tokens'],
        'output_tokens': response['usage']['completion_tokens'],
        }