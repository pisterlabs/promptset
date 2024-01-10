import json
from typing import Dict

import openai

chat_history = [
    {
        "role": "system",
        "content": "You are a REST API server with an endpoint /generate-random-question/:topic/:difficultyLevel, which generates unique random quiz question in json data, difficulty Level are 1 to 10, with 1 being easiest question and 10 being hardest question.",
    },
    {   
        "role": "user",     
        "content": "GET /generate-random-question/general knowledge/3"
    },
    {
        "role": "assistant",
        "content": '\n\n{\n    "question": "What is the smallest country in the world by land area?",\n    "options": ["San Marino", "Maldives", "Monaco", "Vatican City"],\n    "answer": "Vatican City",\n    "explanation": "Vatican City is the smallest country in the world by land area with an area of approximately 44 hectares or 110 acres."\n}',
    },
    {   
        "role": "user", 
        "content": "GET /generate-random-question/general knowledge/2"
    },
    {
        "role": "assistant",
        "content": '\n\n{\n    "question": "What is the capital of Australia?",\n    "options": ["Melbourne", "Sydney", "Canberra", "Brisbane"],\n    "answer": "Canberra",\n    "explanation": "Canberra is the capital city of Australia. It is located in the southeastern part of the country and is home to many important government institutions and cultural sites."\n}',
    },
]


def get_quiz_questions(difficultyLevel: int) -> Dict[str, str]:
    global chat_history
    openai.api_key = 'api-key'
    current_chat = chat_history[:]
    current_user_message = {
        "role": "user",
        "content": f"GET /generate-random-question/general knowledge/{difficultyLevel}",
    }
    current_chat.append(current_user_message)
    chat_history.append(current_user_message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=current_chat,
        n=5
    )
    quiz = []
    for i in 0, 1, 2, 3, 4:
        try: 
            quiz.append(json.loads(response["choices"][i]["message"]["content"]))
            current_assistent_message = {"role": "assistant", "content": response["choices"][i]["message"]["content"]}
            chat_history.append(current_assistent_message)
        except:
            continue
    return quiz
