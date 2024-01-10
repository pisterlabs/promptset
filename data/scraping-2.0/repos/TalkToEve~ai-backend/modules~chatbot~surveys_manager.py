from configurations.config_llm import CAMBALACHE_TEMPERATURE
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import os
import sys

import random
import pandas as pd
import json

# Obtain the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
# Add the path to the sys.path list
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from configurations.config_surveys import SURVEYS_PATH

class BaseSurveysManager():
    def __init__(self,):
        pass
    
    def load_survey(self, survey):
        raise NotImplementedError
    
    def obtain_questions(self,):
        raise NotImplementedError
    
    def obtain_answers(self,):
        raise NotImplementedError
    
    def load_responses(self,question_id=None, question = None, answer = None):
        raise NotImplementedError
    
class SurveyManager_v1(BaseSurveysManager):
    def __init__(self,):
        self.surveys_dataframe = None
        self.path_to_lists_surveys = ""
        
        self.survey_name = ""
        self.survey_date = ""
        self.survey_answered = False
        
        self.survey = {}
        self.survey_responses = {}
        self.path_to_survey_folder = ""
        
    def load_list_of_surveys(self,path):
        # CHeck if the path is a csv file
        if not path.endswith(".csv"):
            raise Exception("The path must be a csv file")
        
        # Read the path 
        self.path_to_lists_surveys = path
        # Read the csv file
        self.surveys_dataframe = pd.read_csv(self.path_to_lists_surveys)
        
    def select_survey_to_answer(self,):
        # Obtain the lists of surveys that not was answered
        surveys_not_answered = self.surveys_dataframe[self.surveys_dataframe['answered'] == False].values.tolist()
        
        # Generate a random number to select the survey to answer
        if len(surveys_not_answered) == 0:
            survey_to_answer = None
        elif len(surveys_not_answered) == 1:
            survey_to_answer = surveys_not_answered[0]
        else:
            survey_to_answer = random.choice(surveys_not_answered)
        
        if survey_to_answer:
            self.survey_name = survey_to_answer[0]
            self.survey_date = survey_to_answer[1]
            self.survey_answered = survey_to_answer[2]
          
    def check_if_has_survey_to_answer(self,):
        # Check if there is a survey to answer
        if len(self.surveys_dataframe[self.surveys_dataframe['answered'] == False]):
            return True
        else:
            return False
        
    def load_survey(self, path_to_load_survey=None, survey_id = None):
        if survey_id is None:
            # Check that self.survey_name and self.survey_date are not empty
            if self.survey_name == "" or self.survey_date == "":
                raise Exception("The survey must have a name and a date")
        
        # Checj that path_to_load_survey exists
        if path_to_load_survey is None:
            path_to_load_survey = SURVEYS_PATH
            
        if not os.path.exists(path_to_load_survey):
            raise Exception("The path to load the survey does not exist")
        
        if survey_id is None:
            survey_id = self.survey_date + '_' + self.survey_name
            
        self.path_to_survey_folder = os.path.join(path_to_load_survey, survey_id)
        
        # Read the json file
        with open(os.path.join(self.path_to_survey_folder, survey_id + '.json')) as json_file:
            self.survey = json.load(json_file)
    
    def obtain_number_of_questions(self,):
        return len(self.survey) - 3
    
    def obtain_question(self, number_question, return_dict = False):
        # Check that number_question is a number
        if not isinstance(number_question, int):
            raise Exception("The number of the question must be a number")
        # Check that number_question is a valid number
        if number_question >= len(self.survey) - 2:
            raise Exception("The number of the question must be a valid number")
        
        if return_dict:
            # Obtain the question
            return self.survey['question_' + str(number_question)]
        else:
            # Obtain the question
            question = self.survey['question_' + str(number_question)]['question']
            answers = self.survey['question_' + str(number_question)]['answers']
            return question, answers
    
    def load_response(self, number_question, response):
        # Check that number_question is a number
        if not isinstance(number_question, int):
            raise Exception("The number of the question must be a number")
        # Check that number_question is a valid number
        if number_question >= len(self.survey) - 2:
            raise Exception("The number of the question must be a valid number")
        # Check that response is a str
        if not isinstance(response, str):
            raise Exception("The response must be a string")
        
        # Save the response in the survey_responses
        question = self.survey['question_' + str(number_question)]['question']
        self.survey_responses[question] = [response]
            
    def save_survey_responses(self,):
        # Path to save results in dataframe
        survey_name = self.survey_date + '_' + self.survey_name
        path_to_result = os.path.join(SURVEYS_PATH, survey_name, 'results' + '_' + survey_name + '.csv')
        
        # Now we need to load the dataframe
        df = pd.read_csv(path_to_result)
        
        # Now we need to convert survey_responses in a dataframe
        survey_responses_dataframe = pd.DataFrame(self.survey_responses, index=[0])
        
        # Now we need to concatenate the two dataframes
        df = pd.concat([df, survey_responses_dataframe], axis=0, ignore_index=True)
        
        # Now we need to randomly the rows of the dataframe
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Now we save the dataframe
        df.to_csv(path_to_result, index=False)
        
        # Now we need to change the survey dataframe
        self.surveys_dataframe.loc[self.surveys_dataframe['survey_name'] == self.survey_name, 'answered'] = True
        
        # Save the new dataframe
        self.surveys_dataframe.to_csv(self.path_to_lists_surveys, index=False)
        