import openai
import os
from scripts import utils
from dotenv import load_dotenv

def assemble_output(question:str, target_schema:str, baseline_sql:str, qa_str:str):
    """
    Function to assemble final sql from LLM
    """
    # Setup OpenAI API
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = utils.get_model()

    # Build prompt
    messages = []

    # Build system instruction
    system_instruction = {
        "role": "system",
        "content": "You will be given a \"User question\", \"Schema\", a \"Baseline sql query\", and \"Knowledge\" about the schema. Use all this information to respond with an sql query for sqlite that accurately answers the \"User question\". \nImportant: Your response should include the sql query only. No other text should be included in your response."
    }
    messages.append(system_instruction)

    # Build user question
    user_question = {
        "role": "user",
        "content": "User question:\n" + question + "\n\nSchema:\n" + target_schema + f"\n\nBaseline sql query: {baseline_sql}\n\n" + "Knowledge:\n" + qa_str
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
        print('[ERROR] OpenAI API call failed for assemble output.')
        return None

    # Clean up final sql string
    response_str = response['choices'][0]['message']['content']
    start_idx = response_str.lower().find('select')
    final_sql = response_str[start_idx:].replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()

    # Return response
    return {
        'final_sql': final_sql,
        'input_tokens': response['usage']['prompt_tokens'],
        'output_tokens': response['usage']['completion_tokens']
    }