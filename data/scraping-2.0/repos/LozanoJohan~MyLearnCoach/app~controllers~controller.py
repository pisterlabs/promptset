import dotenv

from controllers.sia_scrapper import SiaScrapper
from controllers.coursera_scrapper import CourseraScrapper
from models.sia_course import SIACourse
from models.sia_course import Group
from models.coursera_course import CourseraCourse

import json
import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from pathlib import Path



# Ruta absoluta de el archivo con los datos
json_path = Path(__file__).resolve().parent.parent / 'data' / 'courses_data.json'



class Controller:
    def __init__(self):

        dotenv.load_dotenv()
        os.getenv('OPENAI_API_KEY')

        self.set_llm()
    
    def set_api_key(self, api_key):
        os.environ['OPENAI_API_KEY'] = api_key
    
    def fetch_sia_courses(self):
        sia = SiaScrapper()
        courses = sia.scrap()

        # Write courses to JSON file
        with open('sia_courses.json', 'w') as f:
            json.dump(courses, f)
            f.close()

        return courses
    
    def get_data(self):
        # Read from json file
        with open(json_path, "r") as file:
            data = json.load(file)

        return data
    
    def set_llm(self):
        
        template = """Question: {question}

        Respuesta:."""

        self.title_template = PromptTemplate(
            input_variables = ['topic'], 
            template='{topic}'
        )

        self.script_template = PromptTemplate(
            input_variables = ['title'], 
            template = 'Escribe la o las palabras principales para poder buscar acerca de este tema: {title}'
        )

        #self.llm = GPT4All(model=local_path, verbose=True)
        self.llm = OpenAI(verbose=True, temperature=0.7)

        self.title_chain = LLMChain(llm = self.llm, prompt = self.title_template, verbose = True, output_key = 'title')
        self.script_chain = LLMChain(llm = self.llm, prompt = self.script_template, verbose = True, output_key = 'script')
        
        self.sequential_chain = SequentialChain(chains = [self.title_chain, self.script_chain], 
                                                input_variables = ['topic'], 
                                                output_variables = ['title', 'script'], 
                                                verbose = True)
    
    def process_input(self, input):

        response = self.sequential_chain({'topic': input})
        return response
    

    def get_sia_courses(self, query_type, query):
        courses = []

        # Read from json file
        with open(json_path, "r") as file:
            data = json.load(file)

        query_parser = {'Nombre':'name','Código':'code','----':'default'}


        if query_parser[query_type] == 'code':

            courses = [course_data for course_data in data.values() if course_data['code'] == query]


        elif query_parser[query_type] == 'name':

            courses = [course_data for course_data in data.values() if course_data['name'] == query]


        elif query_parser[query_type] == 'default':

            courses = [course_data for course_data in data.values()]


        # for id, course_data in data.items():

        #     if query_parser[query_type] == 'default':
        #         groups = []

        #         if 'groups' not in course:
        #             course['groups'] = ''

        #         for group_data in course['groups']:
        #             group = Group( **group_data )
        #             groups.append(group)

        #         course =  SIACourse(groups=groups, **course)                    
        #         courses.append(course)
                
        #     elif course[query_parser[query_type]] == query:
        #         groups = []

        #         if 'groups' not in course:
        #             course['groups'] = ''

        #         for group_data in course['groups']:
        #             group = Group( **group_data )
        #             groups.append(group)

        #         course =  SIACourse(course['name'], 
        #                                 groups, 
        #                                 "course['description']", 
        #                                 course['code'], 
        #                                 course['credits'], 
        #                                 course['type'])                    
        #         courses.append(course)
            
        return courses
    
    def get_coursera_courses(self, query:str):
        courses = []
        #import streamlit as st

        query = query.replace('"', '').lower()
        try:
            _ = query.split(':')
            #st.write(_[1])
            keywords = _[1].split(',')
        except:
            keywords = query.split(',')
        
        #for keyword in keywords:
            # st.write(keyword)
            # st.write(keyword in 'Introduction to Data Science')
        # Read from json file

        for keyword in keywords:

            courses.append(CourseraScrapper.scrap(query=keyword))
        
        return courses, keywords

        # for course_data in data['CourseraCourses']:

        #     for keyword in keywords:
                
        #         # st.write(course_data['name'].lower())
        #         # st.write('Keword:', keyword, 'Nombre:', course_data['name'].lower(), keyword in course_data['name'].lower())
        #         if keyword in course_data['name'].lower():

        #             course = CourseraCourse( **course_data )
        #             courses.append(course)

        #             #st.markdown(f"**{course}**")
            
            
        # return courses, keywords

