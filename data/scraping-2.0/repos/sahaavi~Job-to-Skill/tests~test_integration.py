import requests
import openai
import pandas as pd
import sys
import os
# sys.path.append('/Users/nomanmohammad/Desktop/Job-to-Skill')

from job_skill import job_viz as jv
from job_skill import openai_api as oa

import unittest
from dotenv import load_dotenv

class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()
        print("integration test starts")
        cls.api_key = os.getenv("API_KEY")
        # cls.api_key = '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'

        cls.url = 'https://ca.linkedin.com/jobs/view/lead-software-engineering-decision-management-platform-at-mastercard-3378359650'

        cls.df = pd.DataFrame({'Job URL': ['https://www.linkedin.com/jobs/view/3472248063'], 
        'Tools': ['Azure/AWS, MLOps, Data Management and Analytics Platforms'],
        'Programming Languages': ['Java, C#, Ruby, Python'],
        'Job Location': 'Toronto, ON, Canada'})

        cls.tools = pd.DataFrame({'Tools': ['AWS', 'Azure', 'Excel']})
        cls.df_lang = pd.DataFrame({'Programming Languages': ['Java', 'C#', 'Ruby', 'Python']})
        cls.job_skills = [['python', 'SQL', 'java']]
        cls.interview_questions = ['What Big Data Technologies have you worked with?', 'What strategies have you used to capture and develop ideas?']

        cls.job_description = """Understand the day-to-day issues that our business faces, which can be better understood with data
            Collecting and analyzing data from a variety of sources (such as college and university websites, government databases, and industry reports)
            Cleaning, formatting, and organizing the data in a way that is useful and easy to understand
            Creating Excel sheets or other data visualization tools to display the information in a clear and visually appealing way
            Identifying trends and patterns in the data, and providing insights and recommendations based on the analysis
            Collaborating with other teams within the company to ensure that the data is accurate and relevant to their needs
            Continuously monitoring and updating the data set to ensure its relevance and accuracy
            Compile and analyze data related to business' issues. Develop clear visualizations to convey complicated data in a straightforward fashion."""

    # setting up for test
    def setUp(self):
        print("Test Setup")

    # test end
    def tearDown(self):
        print("Test End")

    @classmethod
    def tearDownClass(cls):
        print("integration test finishes")

    # testing integration between scrape_job_description and requests module for proper response
    def test_integration_scrape_job_description(self):
        self.assertEqual(requests.get(self.url).status_code, 200)

    # testing integration between the parse_df function and visualize_info
    def test_integration_visualize_info_bar(self): 
        df_tools2 = jv.parse_df(self.df, 'Tools')
        df_lang2 = jv.parse_df(self.df, 'Programming Languages')
        self.assertEqual(jv.visualize_info(df_lang2, df_tools2).to_dict()['vconcat'][0]['mark']['type'], 'bar')

    # testing integration between the parse_df function and visualize_location
    def test_integration_visualize_location(self): 
        self.assertEqual(jv.visualize_location(self.df, 'Job Location').to_dict()['layer'][0]['mark']['type'], 'text')

    # testing integration between openAI API and 'get_questions' 
    def test_integration_get_questions(self):

        openai.api_key = self.api_key
        pre_prompt = """Given the list of skills:"""
        post_prompt = """Return 5 relevant interview questions."""
        prompt = pre_prompt + str(self.job_skills[0]) + post_prompt

        response_json = openai.Completion.create(
            engine="text-davinci-003",
            temperature=0.2,
            prompt=prompt,
            max_tokens=200)

        self.assertIsInstance(oa.get_questions(response_json), list)

    # testing integration between openAI API and 'get_question_answers' 
    def test_integration_get_question_answers(self):
        openai.api_key = self.api_key
        pre_prompt = """Given the list of potential data science interview questions: """
        post_prompt = """Return a relevant response to each interview question in the following form: 1. response2. response3. response4. response5. response"""
        prompt = pre_prompt + str(self.interview_questions) + post_prompt

        response_json = openai.Completion.create(
            engine="text-davinci-003",
            temperature=1,
            prompt=prompt,
            max_tokens=1000)

        self.assertIsInstance(oa.get_question_answers(response_json), list)

    # testing integration between openAI API and 'get_skills' 
    def test_integration_get_skills(self):
        openai.api_key = self.api_key
        pre_prompt = """Based on the job description identify the required skills in percentage of 100. 
                Job Description:
                """
        post_prompt = """
                Give the answer like this-
                Skill: Percentage:"""
        prompt = pre_prompt + self.job_description + post_prompt

        response_json = openai.Completion.create(
            engine="text-davinci-003",
            temperature=0.1,
            prompt=prompt,
            max_tokens=256)

        self.assertIsInstance(oa.get_skills(response_json), list)
        

unittest.main(argv=[''], verbosity=2, exit=False) 
