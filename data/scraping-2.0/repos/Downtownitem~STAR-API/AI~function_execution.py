import os
import json
from typing import List, Dict
import openai
import threading

from Database.Vectorial.vector_database import FullControl

openai.api_key = os.getenv('OPENAI_API_KEY')

control = FullControl()


def get_one_response(text: List[Dict[str, str]]):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=text,
        max_tokens=500
    )

    return response['choices'][0]['message']['content']


def get_suggestions(text):
    try:
        text = get_one_response([
            {
                'role': 'user',
                'content': """Answer just with the following json format:
     {
        "suggested_process": "<Recommended procedure>", (Required)
        "process_steps": [<Step 1>, <Step 2>, ...], (Required)
        "extra": [<Extra suggestion>, ...] (Not required)
    }
    
    Create suggestions that are related and that can help to complete this task/answer this question:\n""" + text
            }
        ])

        text = text[text.find('{'):text.rfind('}') + 1]
        json_text = json.loads(text)

        if 'suggested_process' not in json_text or 'process_steps' not in json_text:
            return {'suggested_process': 'No suggestions found', 'process_steps': [], 'extra': []}
        else:
            return json_text
    except:
        return {'suggested_process': 'No suggestions found', 'process_steps': [], 'extra': []}


def get_secondary_questions(text):
    try:
        text = get_one_response([
            {
                'role': 'user',
                'content': """Answer with the following json format:
 {
    "related_questions": [<Question>, <Question>, ...]
}

In case it is a text create 3 questions that are related to it, in case it is a question create 3 related questions with secondary doubts that can help to answer the main one:\n""" + text
            }
        ])

        text = text[text.find('{'):text.rfind('}') + 1]
        json_text = json.loads(text)

        if 'related_questions' not in json_text:
            return {'related_questions': []}
        else:
            return json_text
    except:
        return {'related_questions': []}


def get_information_about(question, suggestion):
    final_response = {}

    def get_suggestion_thread(suggestion):
        suggestion = get_suggestions(suggestion)
        final_response['suggestions'] = suggestion

    if suggestion is not None:
        yield {'type': 'function', 'content': 'Creating for suggestions'}
        suggestion_thread = threading.Thread(target=get_suggestion_thread, args=(suggestion,))
        suggestion_thread.start()

    if question is not None:
        yield {'type': 'function', 'content': 'Creating related questions'}

        secondary = get_secondary_questions(question)

        secondary_solutions = []
        for i in secondary['related_questions']:
            yield {'type': 'function', 'content': 'Searching for ' + str(i)}
            temp = {
                'question': i,
                'search_result': control.query(i, top_k=1),
            }

            secondary_solutions.append(temp)
        final_response['related_questions'] = secondary_solutions

    if suggestion is not None:
        suggestion_thread.join()

    yield {'type': 'function', 'content': 'Searching for ' + question}
    final_response['main_question'] = {
        'question': question,
        'search_result': control.query(question, top_k=1),
    }

    yield {'type': 'message', 'content': final_response}
