import json
import os
import datetime
from abc import ABC, abstractmethod
from typing import Any
from gtts import gTTS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from langchain.chains import LLMChain, ConversationChain
import openai
from langchain.chat_models import ChatOpenAI
from chatbot_settings import ChatBotSettings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from pandasai import PandasAI

import pandas as pd






llm = ChatOpenAI(
            temperature=0,
            openai_api_key=ChatBotSettings().OPENAI_API_KEY(),
            model_name="gpt-3.5-turbo"
)


def load_patient_questions():
    # Define the file path
    file_path = 'patient_questions.json'

    # Check if the file exists
    if os.path.exists(file_path):
        # Load questions from the existing file
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
            loaded_questions = loaded_data["questions"]
    else:
        # Put questions into a list
        questions = [
            "How are you feeling today?",
            "Have you experienced any symptoms?",
            "Are you taking any medications?",
            "What is your level of pain? (1-5)",
            "Do you have any other symptoms?"
            # Add more questions here...
        ]

        # Create the JSON object
        data = {
            "questions": questions
        }

        # Save the initial questions to the file
        with open(file_path, 'w') as f:
            json.dump(data, f)

        # Assign the initial questions to loaded_questions
        loaded_questions = questions

    return loaded_questions

# Call the function to load patient questions
patient_questions = load_patient_questions()
# Print the loaded questions
print(patient_questions)


patients = [{"name": "John Doe"}, {"name": "Jane Doe"}, {"name":"Steve Smith"}]  # List of patients

responses = []

for patient in patients:
    
    patient_responses = {
        "name": patient["name"],
        "date": str(datetime.date.today()),
        "responses": {},
        "questions": []
    }
    questions = patient_questions

    print('Hi ' + patient["name"])
    # Predefined questions
    for question in questions:
        answer = input(question + " ")
        patient_responses["responses"][question] = answer

    
    question_response_pairs = [f"{question}: {response}" for question, response in patient_responses['responses'].items()]
    questions_and_responses = ' '.join(question_response_pairs)

    template = """The following is a response from an AI Chiropractor. The AI Chiropractor has an excellent bedside manner provides specific details from its context. 
    If the AI does not know the answer to a question, it truthfully says it does not know. The AI does notd emands that someone asks a question.
    {history}
    Patient: {input} """ + "Patient: " + questions_and_responses

    prompt = PromptTemplate(
            input_variables=["history","input"], template=template
    )
    
    conversation_with_chain = ConversationChain(
        llm=llm, 
        verbose=True, 
        prompt=prompt,
        memory=ConversationBufferMemory()
    )

    # Open-ended questions
    while True:
        question = input(patient["name"] +  ", What are your questions? (type 'done' when you are finished) ")
        if question.lower() == 'done':
            break

        answer = conversation_with_chain(question)
        print(answer['response'])

        patient_responses["responses"][question] = answer
        patient_responses["questions"].append(question)

    responses.append(patient_responses)

# Load existing responses from the file
with open('patient_responses.json') as f:
    existing_responses = json.load(f)

# Append the new responses to the existing ones
existing_responses.extend(responses)

# Save the updated responses back to the file in JSON format
with open('patient_responses.json', 'w') as f:
    json.dump(existing_responses, f)





