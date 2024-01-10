import json
import logging
import openai
import os

from dotenv import load_dotenv, find_dotenv
from copy import deepcopy
from util import PipelineStep

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.getenv('OPEN_AI_API_KEY')


class RerankOpenaiPipelineStep(PipelineStep):

    def __init__(self, name: str, enabled: bool = True):
        super().__init__(name, enabled=enabled)
        self.logger = logging.getLogger("reorder")

    def execute_step(self, input_data):
        if not self.enabled:
            self.logger.info(f"Step {self.name} is disabled.")
            return input_data

        # Define the query and candidate results
        query = input_data["search_text"]
        query_results = input_data["result_items"]

        rerank_prompt = f"""
        Your task is to rerank the provided sentences in an array using a query. The structure of the array is between
        the next back ticks"""
        rerank_prompt += """
        ```
        [{'id': 1, 'name': 'The sentence'}]
        ```
        """
        rerank_prompt += f"""
        Return the results as a json object, that includes an array with the re-ordered objects. Add the explanation to 
        the json object using the property 'explanation' for each object.
        
        sentences: {query_results}
        query: {query}
        """

        response = self.__call_openai(rerank_prompt)

        output_data = deepcopy(input_data)
        output_data["result_items_reranked_openai"] = response
        return output_data

    def __call_openai(self, prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message["content"]
        return json.loads(content)
