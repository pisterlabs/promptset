# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:29:59 2023

@author: abiga
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:15:02 2023

@author: abiga
"""
import os
from enum import Enum
import openai
from marvin import ai_classifier, ai_fn
import logging
from marvin import settings

#logging.basicConfig(level=logging.DEBUG) # if you want to see the JSON getting passed to OpenAI
logging.disable(logging.CRITICAL) # if you want to turn off the logging
settings.llm_request_timeout_seconds = 6000

llm_model='openai/gpt-3.5-turbo'


logging.basicConfig(level=logging.DEBUG) # if you want to see the JSON getting passed to OpenAI
#logging.disable(logging.CRITICAL) # if you want to turn off the logging

openai.api_key = os.environ.get("OPENAI_API_KEY")



@ai_fn
def extract_entities_one(duties: str) -> list[str]:
    """
    Given `duties`, extract just the names of software tools and programming languages, and return these in a list. Return empty list if there are none.
    """  
    
@ai_fn
def extract_entities_two(duties: str) -> list[str]:
    """
    Given `duties`, performs Named Entity Recognition, filters to just entities which are software tools or programming languages, and returns list of results.
    """

@ai_fn
def extract_entities_three(duties: str) -> list[str]:
    """
    Given `duties`, extract just the names of specific, named software tools and programming languages, and return these in a list. Return empty list if there are none.
    """  
    
    
@ai_fn
def extract_entities_four(duties: str) -> list[str]:
    """
    The programming languages identified in the given string. Return empty list if there are none.
    """  

@ai_fn
def extract_entities_five(duties: str) -> list[str]:
    """
    The names of software tools identified in the given string. Return empty list if there are none.
    """  

    

class JobAnalyzer:
    def __init__(self, duties: str):
        self.duties = duties
        self.entities_one = extract_entities_one(self.duties)
        self.entities_two = extract_entities_two(self.duties)
        self.entities_three = extract_entities_three(self.duties)
        self.entities_four = extract_entities_four(self.duties)
        self.entities_five = extract_entities_five(self.duties)

    def __repr__(self):
        return f"Duties: {self.duties[:100]}...\n\{self.entities_one}\n\{self.entities_two}\n\{self.entities_three}\n\{self.entities_four}\n\{self.entities_five}"


def test_functions():
    # List of duties for testing
    duties_samples_list = [
    "Responsible for designing algorithms using Python. Experience with TensorFlow and PyTorch required. Collaborate using Jira and Git.",

    "Handle office administration, schedule meetings, and manage office supplies.",

    "Works with the Department of State in Arlington, VA on big problems",
    
    "Does math, data analysis, and programming. Generates algorithms.",
    
    "Makes dashboards and pivot tables in RStudio"
    ""
    ]

    # Test with the provided JobAnalyzer class
    for duties in duties_samples_list:
        analysis = JobAnalyzer(duties)
        print(analysis)
        print("-" * 80)

test_functions()