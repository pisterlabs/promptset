from tasks.abstractTask import AbstractTask
from tenacity import retry, stop_after_attempt, wait_fixed, before
import requests
from bs4 import BeautifulSoup
from requests.exceptions import Timeout
import logging
from openai import OpenAI
import sys

class ScrapperTask(AbstractTask):
    HEADER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    
    def process_task_details(self):
        task_content = self.assignment_body['input']
        task_instruction = self.assignment_body['msg']
        question = self.assignment_body['question']
        try: 
            page_content = self.fetch_article(task_content)
            system_instruction =  task_instruction +  " #### \n " +  page_content
            result = self.process_chat_query(question, system_instruction)
            return result
        except Exception as e:
            logging.error(f'Exiting program due to validation error: {e}')
            sys.exit()
     
    def solve_task(self):
        super().solve_task()

    def before_log(retry_state):
        logging.info(f"Retrying for {retry_state.attempt_number} time(s)")

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(3), before=before_log)
    def fetch_article(self, url: str):
        headers = {
            "User-Agent": self.HEADER
        }
        try:
            response = requests.get(url, headers=headers, timeout=60)
            print(response.status_code)
            if response.status_code != 200:
                raise Exception("Unable to fetch webpage")

            page_content = BeautifulSoup(response.content, 'html.parser')
            return page_content.getText()
        except Timeout as e:
            print(f"Timeout occurred while fetching the webpage: {str(e)}")
            raise e

    def process_chat_query(self, question, system_instruction):
        model = OpenAI()
        completion = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": question}
            ]
        )
        return completion.choices[0].message.content
