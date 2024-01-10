import json
from typing import Dict

import openai

chat_history = [
    {
        "role": "system",
        "content": "You are a REST API server with an endpoint /generate-random-question/:topic, which generates unique random quiz question in json data.",
    },
    {"role": "user", "content": "GET /generate-random-question/devops"},
    {
        "role": "assistant",
        "content": '\n\n{\n    "question": "What is the difference between Docker and Kubernetes?",\n    "options": ["Docker is a containerization platform whereas Kubernetes is a container orchestration platform", " Kubernetes is a containerization platform whereas Docker is a container orchestration platform", "Both are containerization platforms", "Neither are containerization platforms"],\n    "answer": "Docker is a containerization platform whereas Kubernetes is a container orchestration platform",\n    "explanation": "Docker helps you create, deploy, and run applications within containers, while Kubernetes helps you manage collections of containers, automating their deployment, scaling, and more."\n}',
    },
    {"role": "user", "content": "GET /generate-random-question/jenkins"},
    {
        "role": "assistant",
        "content": '\n\n{\n    "question": "What is Jenkins?",\n    "options": ["A continuous integration server", "A database management system", "A programming language", "An operating system"],\n    "answer": "A continuous integration server",\n    "explanation": "Jenkins is an open source automation server that helps to automate parts of the software development process such as building, testing, and deploying code."\n}',
    },
]


def get_quiz_from_topic(topic: str, api_key: str) -> Dict[str, str]:
    global chat_history
    openai.api_key = api_key
    current_chat = chat_history[:]
    current_user_message = {
        "role": "user",
        "content": f"GET /generate-random-question/{topic}",
    }
    current_chat.append(current_user_message)
    chat_history.append(current_user_message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=current_chat
    )
    quiz = response["choices"][0]["message"]["content"]
    current_assistent_message = {"role": "assistant", "content": quiz}
    chat_history.append(current_assistent_message)
    print(f"Response:\n{quiz}")
    return json.loads(quiz)
