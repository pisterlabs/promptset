import os
from dotenv import load_dotenv
from typing import List
import ast
import openai
from dataclasses import dataclass
from pyairtable.api.table import Table


load_dotenv()
AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY', "")
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID', "")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', "")
BEST_PERFORMING_VIDEOS_TABLE = os.getenv('BEST_PERFORMING_VIDEOS_TABLE', "")
RECOMMENDATIONS_TABLE = os.getenv('RECOMMENDATIONS_TABLE', "")


@dataclass
class Recommendations:
    num_titles_to_check: int
    num_titles_to_recommend: int

    def __get_best_peforming_video_titles(self) -> List[str]:
        table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, BEST_PERFORMING_VIDEOS_TABLE)
        records = table.all()
        titles = [record['fields']['Title'] for record in records]
        return titles


    def __generate_recommendations(self, titles: List[str]) -> List[str]:
        def __get_completion(prompt, model="gpt-3.5-turbo"):
            messages= [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=700,
                temperature=0.7,
            )
            return response.choices[0].message["content"]

        prompt = f"""
        You are an expert Youtube video recommender system. 
        Based on your knowledge and on the following video titles, recommend {self.num_titles_to_recommend} great Youtube video titles. The response should be in a Python list format.
        ```{titles}```
        """
        response = __get_completion(prompt)
        print("Recommendations: " + response)
        recommendations = ast.literal_eval(response)
        return recommendations


    def __insert_recommendations_to_airtable(self, recommendations: List[str]):
        table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, RECOMMENDATIONS_TABLE)
        for recommendation in recommendations:
            record = {'Recommendation': recommendation}
            table.create(record)


    def generate(self):
            titles = self.__get_best_peforming_video_titles()
            titles_to_check = titles[:self.num_titles_to_check]
            recommendations = self.__generate_recommendations(titles_to_check)
            self.__insert_recommendations_to_airtable(recommendations)        
