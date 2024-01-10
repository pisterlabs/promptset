from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from datetime import datetime
from langchain.llms import OpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

class HistoryQuiz():
    
    def create_history_question(self,topic):
        '''
        This method should output a historical question about the topic that has a date as the correct answer.
        For example:
        
            "On what date did World War 2 end?"
            
        '''
        # PART ONE: SYSTEM
        system_template="You write single quiz questions about {topic}. You only return the quiz question."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        # PART TWO: HUMAN REQUEST
        human_template="{question_request}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        # PART THREE: COMPILE TO CHAT
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        # PART FOUR: INSERT VARIABLES
        request = chat_prompt.format_prompt(topic=topic,question_request="Give me a quiz question where the correct answer is a specific date.").to_messages()
        # PART FIVE: CHAT REQUEST
        chat = ChatOpenAI()
        result = chat(request)
        
        
        return result.content
    
    def get_AI_answer(self,question):
        '''
        This method should get the answer to the historical question from the method above.
        Note: This answer must be in datetime format! Use DateTimeOutputParser to confirm!
        
        September 2, 1945 --> datetime.datetime(1945, 9, 2, 0, 0)
        '''
        # Datetime Parser
        output_parser = DatetimeOutputParser()
        
        # SYSTEM Template
        system_template = "You answer quiz questions with just a date."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        
        # HUMAN Template
        human_template = """Answer the user's question:
        
        {question}
        
        {format_instructions}"""
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        # Compile ChatTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
        
        # Insert question and format instructions
        
        request = chat_prompt.format_prompt(question=question,
                                            format_instructions=output_parser.get_format_instructions()).to_messages()
        
        
        
        # Chat Bot
        chat = ChatOpenAI()
        result = chat(request)
        # Format Request to datetime
        correct_datetime = output_parser.parse(result.content)
        return correct_datetime
    
    def get_user_answer(self,question):
        '''
        This method should grab a user answer and convert it to datetime. It should collect a Year, Month, and Day.
        You can just use input() for this.
        '''
        print(question)
        

        # Get the year, month, and day from the user
        year = int(input("Enter the year: "))
        month = int(input("Enter the month (1-12): "))
        day = int(input("Enter the day (1-31): "))

        # Create a datetime object
        user_datetime = datetime(year, month, day)

        
        return user_datetime
        
        
    def check_user_answer(self,user_answer,ai_answer):
        '''
        Should check the user answer against the AI answer and return the difference between them
        '''
        
        # Calculate the difference between the dates
        difference = user_answer - ai_answer

        # Format the difference into a string
        formatted_difference = str(difference)

        # Return the string reporting the difference
        print("The difference between the dates is:", formatted_difference)

quiz_bot = HistoryQuiz()
question = quiz_bot.create_history_question(topic='World War 2')
ai_answer = quiz_bot.get_AI_answer(question)
user_answer = quiz_bot.get_user_answer(question)
quiz_bot.check_user_answer(user_answer,ai_answer)