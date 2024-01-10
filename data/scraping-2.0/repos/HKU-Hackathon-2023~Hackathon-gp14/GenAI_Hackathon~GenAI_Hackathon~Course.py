import os
from dotenv import load_dotenv
load_dotenv()
from langchain.schema import AIMessage, HumanMessage
from . import prompt  
from langchain.chat_models import ChatOpenAI
import time

class Course:
    def __init__(self, student_education_level, student_special_education_need, subject: str) -> None:
        """This is the constructor of the course class. The first half include instant variable, while the second half is about constructing the course base on input parameter"""
        # instant variable
        self.subject = subject
        self.course_name: str 
        self.estimated_weeks: int 
        self.topic_list: list
        self.weekly_teaching_schedule = dict()
        self.current_week = 0

        # Constructing the course
        self.generate_weekly_topics(student_education_level, student_special_education_need, subject)
        self.generate_course_name()
        self.generate_topic_teaching_instruction(student_special_education_need)


    def generate_weekly_topics(self, student_education_level, student_special_education_need, subject: str) -> None:
        """This function generate weekly topic for the course"""
        LLM = ChatOpenAI(temperature=0) # Create OpenAI instant 

        response = LLM(prompt.get_weekly_topics_format_instructions(student_education_level, student_special_education_need, subject)).content
        weekly_topics = response.split(",")
        self.topic_list = weekly_topics

        for index in range(0, len(weekly_topics)):
            self.weekly_teaching_schedule[f"week_{index}"] = {
                "Topic": weekly_topics[index],
                "Topic_teaching_instruction": str,
                "chat history": list()
            }
        
        self.estimated_weeks = len(weekly_topics)
        print("Weekly Topic Generated: ", weekly_topics)
        return

    def generate_course_name(self) -> None:
        """This function generate the course name"""
        LLM = ChatOpenAI(temperature=0.3) 
        self.course_name = LLM([HumanMessage(content=f"Give a 3 words course name for this course which include topic {self.topic_list}. The course name must be 3 word only with concise. You must give the course name only")]).content
        print(f"Course name decided: {self.course_name}")
        return

    def generate_topic_teaching_instruction(self, student_special_education_need,):
        """This function generate the teaching instruction for each """
        LLM = ChatOpenAI(temperature=0) 
        for index in range(0, 1):
            response = LLM(prompt.get_topic_checklist_instructions( self.weekly_teaching_schedule[f"week_{index}"])).content
            print(f"week {index} teaching instruction: ", response)
            self.weekly_teaching_schedule[f"week_{index}"]["Topic_teaching_instruction"] = response
            self.weekly_teaching_schedule[f"week_{index}"]["chat history"] = [prompt.get_teaching_instruction(student_special_education_need,self.weekly_teaching_schedule[f"week_{index}"]["Topic_teaching_instruction"])]
            if index == 0:
                self.weekly_teaching_schedule[f"week_{index}"]["chat history"].append(AIMessage(content="Hi, I am your teacher - Anson."))
            # time.sleep(20)
        return 

    def get_course_topic(self) -> list:
        """This function will return the topic list of the course"""
        return self.topic_list
    
    def change_topic(self, topic_name: str) -> bool:
        """This function change the current topic. If fails, False, else True"""
        try:
            self.current_week = self.topic_list.index(topic_name)
            return True
        except:
            return False

    def speak_with_virtual_teacher(self, user_message: str) -> str:
        """This function chat with openai"""
        LLM = ChatOpenAI(temperature=0)
        self.weekly_teaching_schedule[f"week_{self.current_week}"]["chat history"].append(HumanMessage(content=user_message))
        response =  LLM(self.weekly_teaching_schedule[f"week_{self.current_week}"]["chat history"]).content
        self.weekly_teaching_schedule[f"week_{self.current_week}"]["chat history"].append(AIMessage(content=response))

        return response
    
    



 














        













