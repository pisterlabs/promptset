from typing import Dict
import json
from langchain.llms import PromptLayerOpenAI
import promptlayer
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
from dotenv import load_dotenv
load_dotenv()
import multiprocessing

PROMPT_LAYER_API_KEY= os.environ['PROMPT_LAYER_API_KEY']
OPEN_API_KEY=os.environ['OPENAI_API_KEY']
FILE_PATH= 'question_search_phrase_sources.json'

def get_prompts(question:str,correct_answer:str,new_answer:str):
    human_message_prompt = f'''Rate how similar these two answers are. Correct_answer:{correct_answer}, New_answer:{new_answer} .'''
    system_message_prompt = f'''
Compare the two answers provided for this question: {question} and provide output in this json format.

Follow these instructions while providing output.
1. Provide a rating out of five for how similar the two answers are.
2. Output should be in this json format - 
---
    {{
    "answer_similarity": "Rate how similar provided new answer is to provided answer",
    "reason": "Reason for rating the similarity."
    }}
----
'''
    return {'human_message_prompt':human_message_prompt,'system_message_prompt':system_message_prompt}



def get_data_from_file(file_path) -> Dict:
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    json_object = json_data
    # Now you can work with the JSON object
    return json_object

def write_to_file(filepath, file_data):
    with open(filepath, 'w') as file:
        file.write(json.dumps(file_data))

    print(f"Data written to {filepath}")
    return

def rate_similarity():
    promptlayer.api_key = PROMPT_LAYER_API_KEY
    chat = PromptLayerOpenAI(temperature=0.4, openai_api_key=OPEN_API_KEY,verbose=True,
                            streaming=False, pl_tags=["answer-similarity"])
    file_data=[]
    json_object = get_data_from_file(FILE_PATH)
    print('READ ALL FILE DATA')
    try:
        for data in json_object:
            correct_answer = data.get('answer')
            question = data.get('question')
            search_phrase_question = data.get('search_phrase_question')
            new_answer = data.get('search_phrase_answer')
            print(f'Analysing {question}')
            prompts_content = get_prompts(new_answer=new_answer,correct_answer=correct_answer,question=question)
            try:
                resp = chat.predict_messages([HumanMessage(content=prompts_content['human_message_prompt']),
                                              SystemMessage(content=prompts_content['system_message_prompt']),
                                              AIMessage(content='You are an AI assisstant that compares two answers of given question and rates how similar two answers are. You always provide output in json format ')
                                              ])

                message = json.loads(resp.content)
                print('Message ', message)
                file_single_obj = {'question': question, 'original': correct_answer,
                                   'search_phrase': search_phrase_question, 'search_phrase_answer': new_answer,
                                   'answer_similarity_score': message['answer_similarity'], 'answer_similarity_reason': message['reason'],
                                   'original_sources':data.get('sources'),'search_phrase_sources':data.get('search_phrase_sources')}
                file_data.append(file_single_obj)
            except Exception as e:
                file_single_obj = {
                    'question': question,
                    'question_answer': correct_answer,
                    'search_phrase': search_phrase_question,
                    'search_phrase_answer': new_answer
                }
                file_data.append(file_single_obj)
                print(f'Exception Occured. Exception: {e} for {file_single_obj}')
        print('LOOP ENDED')
        filepath ='final_data_file.json'

        with open(filepath, 'w') as file:
            file.write(json.dumps(file_data))

        write_to_file('final_data_file_for_comparision.json', file_data)
    except Exception as e:
        print(f'Exception occured: {e}')
        write_to_file('2_answer_similarity_comparison.json', file_data)


rate_similarity()

