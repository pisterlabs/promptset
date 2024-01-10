import os
import openai
import pandas as pd
from dotenv import load_dotenv
import logging
import re
import multiprocessing
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)

class Translator():
    def __init__(self) -> None:
         self.df = pd.DataFrame()
    
    def prompt_translator(self, question_ls, language_ls):
        """_summary_

        Args:
            question_ls (_type_): _description_
            language_ls (_type_): _description_

        Returns:
            _type_: _description_
        """
        custom_prompts  = f"Please translate the following into {language_ls} : {question_ls}"
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": custom_prompts}]
                )
            return completion
        except:
            logging.error("Got error.")

    def parse_responses(self, answers_ls = []):
        """_summary_

        Args:
            answers_ls (list, optional): _description_. Defaults to [].

        Returns:
            _type_: _description_
        """
        logging.info("Parsing responses...")
        all_results = [answer.choices[0].message.content for answer in answers_ls]
        return all_results
    
    def format_answers_to_dataframe(self, all_results : list):
        """
        This take the responses and convert to dataframe

        Args:
            all_results (list): list of responses

        Returns:
            _type_: pandas.DataFrame()
        """
        logging.info(f"Formatting answers to dataframe...")
        print(f"all_results: {all_results}")
        splitted = [re.split('[\n]', answer.strip()) for answer in all_results]
        split_language_and_answers = [re.split('[:]', x) for split in splitted for x in split]
        self.df['LANGUAGE'] = [item[0] for item in split_language_and_answers]
        self.df['TRANSLATION'] = [item[1] for item in split_language_and_answers]
        return self.df
    
    def split_chat_processing(self,questions_ls : list, language_option : str) -> list:
        """
        Parallelize the translator processing

        Args:
            questions_ls (list): _description_
            language_option (list): _description_

        Returns:
            list: List of answers
        """
        answers_ls =[]
        num_processes = os.cpu_count()
        start = time.time()
        questions_map = [(question, language_option) for question in questions_ls]
        with multiprocessing.Pool(processes=4) as pool:
            for result in pool.starmap(self.prompt_translator, questions_map):    
                answers_ls.append(result)
        end = time.time() - start
        logging.info(f"Elapsed processing time:{end}")
        return answers_ls
