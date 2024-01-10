## Chat functions for the app
import requests
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_to_dict
import openai
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    #AIMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser

load_dotenv()


# Set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

# Define a pydantic object to format the response from the model
class TeacherResponse(BaseModel):
    text: str = Field(description="The text response from the teacher")
    #difficulty_score: int = Field(description="The difficulty score of the question")
    #is_correct: bool = Field(description="Whether the user's answer is correct or not")

output_parser = PydanticOutputParser(pydantic_object=TeacherResponse)


# Define a ChatMessage class to handle formatting the messages
class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


# Define a class to handle the chatbot using OpenAI and langchain
class ChatService:
    # Initialize the chatbot with the initial message depending on the context
    def __init__(self):
        # Initialize the chat history
        self.chat_history = ChatMessageHistory()
        # Initialize the chat history dictionary
        self.chat_history_dict = {}
        # Initialize the teacher selecteds
        self.teacher = ""
        # Initialize the topic selected
        self.topic = ""
        # Initialize the age of the student
        self.age = ""
        # Initialize the name of the student
        self.kid_name = ""
        # Initialize the chat_messages
        self.chat_messages = []
      
    
    # Save the chat_history to a dictionary
    def save_chat_history_dict(self):
        self.chat_history_dict = messages_to_dict(self.chat_history.messages)
        return self.chat_history_dict
    
    def add_user_response(self, response: str):
        self.chat_history.add_user_message(response)
        # Return the latest message in the chat history
        chat_messages =  self.save_chat_history_dict()
        # Get the latest message
        latest_message = chat_messages[-1]
        # Convert the response to a Message object
        try:
            formatted_message = ChatMessage(role = "user", content = latest_message[0]['data']['content'])
        except:
            formatted_message = ChatMessage(role = "user", content = latest_message['data']['content'])
        if formatted_message:
            #Append the message to the chat history
            st.session_state.chat_messages.append({"role": formatted_message.role, "content": formatted_message.content})
            # Return the chat history
            return st.session_state.chat_messages
    
    
    def get_initial_message_from_teacher(self):
        messages = [
            {
                "role": "system",
                "content": f"You are a teacher that is helping a child learn about different topics based on role playing games.\
                    You have taken on the personality of {self.teacher} from the kids show 'Paw Patrol' in order to do so.  The child's name is {self.kid_name}.\
                    The topic you are teaching is {self.topic}.  The age of the child is {self.age}.  Please start the conversation off by asking the\
                    child a question to initiate the gameplay back and forth.  This is only one question.  The child will then respond with an answer.  Please only\
                    return the first question, and not a simulated conversation."
            },
        ]

        models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        for model in models:
            for _ in range(3):  # Retry 3 times
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        max_tokens=750,
                        frequency_penalty=0.5,
                        presence_penalty=0.5,
                        temperature=1,
                        top_p=0.9,
                        n=1,
                    )
                    response = response.choices[0].message.content
                    if response:
                        st.session_state.chat_messages.append({"role": "ai", "content": response})
                        return response
                except (requests.exceptions.RequestException, openai.error.APIError):
                    continue  # Skip to the next iteration if an exception is caught
            break  # Break the outer loop if no exception was caught in the inner loop

    def add_message(self, role: str, content: str):
        formatted_message = {"role": role, "content": content}
        st.session_state.chat_messages.append(formatted_message)
        return st.session_state.chat_messages
        
        
    def get_answer_from_teacher(self, kid_response: str):
        st.session_state.chat_messages.append({"role": "user", "content": kid_response})
        prompt = PromptTemplate(
            template = """
        )           You are a teacher that is helping a child learn about different topics by engaging them in an exciting role playing adventure.
                    You have taken on the personality of {teacher} from the kids show 'Paw Patrol'.  The child's name is {kid_name}.
                    The topic you are teaching is {topic}.  The age of the child is {age}.  The format of the conversation is question / response,
                    where you ask a question and the child gives a response.  The conversation should be an engaging and exciting
                    adventure for the child that is fun, educational, and age approrpiate.  Generate your response based on the child's response {kid_response}
                    to your previous question {last_question}.   The conversation history so far is {chat_messages} for reference.
                    Please continue to guide the child on the learning adventure until they are ready to stop.
                    Your response should be a JSON object in this format: {format_instructions}
                    """,
                    input_variables=["kid_response", "teacher", "kid_name", "topic", "age", "chat_messages", "last_question"],
                    partial_variables={"format_instructions" : output_parser.get_format_instructions()}
        )
        system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
        human_template = """Please respond to my answer in the format requested"""
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        messages = chat_prompt.format_prompt(last_question = st.session_state.chat_messages[-1], kid_response = kid_response, teacher = self.teacher, topic = self.topic, age = self.age, kid_name = self.kid_name, chat_messages = st.session_state.chat_messages).to_messages()

        models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k"]
        for model in models:
            try:
                chat = ChatOpenAI(model_name = model, temperature = 1, max_retries=3)

                teacher_response = chat(messages).content

                parsed_response = output_parser.parse(teacher_response)

                # Add the question to the chat history
                st.session_state.chat_messages.append({"role": "ai", "content": parsed_response.text})
                # Check to see whether or not the user got the question correct
                
                return parsed_response.text
                
            except (requests.exceptions.RequestException, openai.error.APIError):
                continue
            


    # Define a function to clear the chat history
    def clear_chat_history(self):
        self.chat_history.clear()
        self.chat_history_dict = {}
        self.chat_messages = []

    
# Define the session variables
session_vars = [
    'chat_session',
]
default_values = [
    ChatService(),
]

# Loop through the session variables and set them if they are not already set
for session_var, default_value in zip(session_vars, default_values):
    if session_var not in st.session_state:
        st.session_state[session_var] = default_value

