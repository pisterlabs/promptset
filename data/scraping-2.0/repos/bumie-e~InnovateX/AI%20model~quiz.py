# get_quiz.py
import json
from typing import Dict
from decouple import config
import openai


openai.api_type = "azure"
openai.api_base = config("AZURE_OPENAI_ENDPOINT") 
openai.api_version = "2023-05-15"
openai.api_key = config("AZURE_OPENAI_KEY")
model_name = 'chatlang'

# I have some chat history saved in a list, where each item is a dictionary representing a message with a role and content.
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

# I define a function that takes a topic string and an API key, and returns a dictionary with a quiz question, options, answer, and explanation.
def get_quiz_from_topic(topic: str) -> Dict[str, str]:
    global chat_history


    # I make a copy of the chat history and add the user's message requesting a quiz question for the given topic.
    current_chat = chat_history[:]
    current_user_message = {
        "role": "user",
        "content": f"GET /generate-random-question/{topic}",
    }
    current_chat.append(current_user_message)
    chat_history.append(current_user_message)

    # I use the OpenAI API to generate a response based on the current chat history.
    response = openai.ChatCompletion.create(
            engine=model_name,
            messages= current_chat, 
            max_tokens=300,
            temperature=0
        )

    # I extract the quiz question from the response and add it to the chat history as an assistant message.
    quiz = response["choices"][0]["message"]["content"]
    current_assistent_message = {"role": "assistant", "content": quiz}
    chat_history.append(current_assistent_message)

    # I print the quiz question and return it as a dictionary.
    print(f"Response:\n{quiz}")
    return json.loads(quiz)