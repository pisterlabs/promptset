# get_quiz.py
import json
from typing import Dict
from decouple import config
import openai
#import secrets_beta

openai.api_type = "azure"
openai.api_base = config("AZURE_OPENAI_ENDPOINT") 
openai.api_version = config("OPENAI_API_VERSION") 
openai.api_key = config("AZURE_OPENAI_KEY")
model_name = config("model_name")

# I have some chat history saved in a list, where each item is a dictionary representing a message with a role and content.

# I define a function that takes a topic string and an API key, and returns a dictionary with a quiz question, options, answer, and explanation.
def get_quiz_from_topic(topic: str) -> Dict[str, str]:
    chat_history = """ You are a REST API server with an endpoint /generate-random-question/:topic, that generates unique random quiz questions in json data.

For example, regarding the request: "GET /generate-random-question/devops"

You can generate the following quiz questions on the topic "devops":

[ {    "content": '\n\n{\n    "question": "What is the difference between Docker and Kubernetes?",\n    "options": ["Docker is a containerization platform whereas Kubernetes is a container orchestration platform", " Kubernetes is a containerization platform whereas Docker is a container orchestration platform", "Both are containerization platforms", "Neither are containerization platforms"],\n    "answer": "Docker is a containerization platform whereas Kubernetes is a container orchestration platform",\n    "explanation": "Docker helps you create, deploy, and run applications within containers, while Kubernetes helps you manage collections of containers, automating their deployment, scaling, and more."\n}',
    },

{
        "content": '\n\n{\n    "question": "Which of the following is NOT a benefit of Infrastructure as Code (IaC)?",\n    "options": ["Consistent environment setup", "Automated infrastructure provisioning", "Reduced manual errors", "Dependence on physical hardware"],\n    "answer": "Dependence on physical hardware",\n    "explanation": "Infrastructure as Code (IaC) allows infrastructure to be provisioned and managed using code and software development techniques. It reduces manual errors, provides consistent environment setup, and enables automated provisioning. It does not, however, introduce a dependence on physical hardware."\n}'
    }
]. """+ f"Now here's a request 'GET /generate-random-question/{topic}'. Generate 5 different unique random quizes for this topic: {topic}"


    current_user_message = [{
        "role": "system",
        "content": chat_history,
    }]

    # I use the OpenAI API to generate a response based on the current chat history.
    response = openai.ChatCompletion.create(
            engine=model_name,
            messages= current_user_message, 
            max_tokens=1098,
            temperature=0
        )

    # I extract the quiz question from the response and add it to the chat history as an assistant message.
    quiz = response["choices"][0]["message"]["content"]
    #current_assistent_message = {"role": "assistant", "content": quiz}
    #chat_history.append(current_assistent_message)

    # I print the quiz question and return it as a dictionary.
    print(f"Response:\n{quiz}")
    json_string = json.dumps([quiz])
    return json.loads(json_string)

def quiz_response(topic):
    quizes = []
    for i in range(5):
        quiz = get_quiz_from_topic(topic)
        #current_assistent_message = {"role": "assistant", "content": quiz}
        #chat_history.append(current_assistent_message)
        quizes.append(quiz)

        # I print the quiz question and return it as a dictionary.
    print(f"Response:\n{quizes}")
    return json.loads(quizes)

