import json
import os
import sqlite3
import sys

import openai
from colorama import Fore

from termy import TERMY_DIR
from termy.constants import GPT3_API_KEY_INPUT, GPT3_CONFIG
from termy.utils import apply_color_and_rest


class GPT3TerminalClient:

    def __init__(self):
        self.API_KEY = self.load_api_key_from_config()
        if not self.API_KEY:
            self.API_KEY = input(GPT3_API_KEY_INPUT)

        if not self.API_KEY:
            sys.exit(apply_color_and_rest(Fore.RED, "No API KEY was provided. Exiting."))

        self.save_api_key_in_config()
        self.platform = self.get_platform()
        self.initDB()

    def initDB(self):
        self.cache = sqlite3.connect(TERMY_DIR / ".cbot_cache")
        self.cache.execute("""
                       CREATE TABLE IF NOT EXISTS questions 
                       (id INTEGER PRIMARY KEY,
                       question TEXT,
                       answer TEXT,
                       count INTEGER DEFAULT 1)""")

    def closeDB(self):
        self.cache.commit()
        self.cache.close()

    def checkQ(self, question_text):
        sql = "SELECT id,answer,count FROM questions WHERE question =" + question_text
        answer = self.cache.execute("SELECT id,answer,count FROM questions WHERE question = ?", (question_text,))
        answer = answer.fetchone()
        if (answer):
            response = answer[1]
            newcount = int(answer[2]) + 1
            counter = self.cache.execute(" UPDATE questions SET count = ? WHERE id = ?", (newcount, answer[0]))
            return (response)
        else:
            return (False)

    def insertQ(self, question_text, answer_text):
        answer = self.cache.execute("DELETE FROM questions WHERE question = ?", (question_text,))
        answer = self.cache.execute("INSERT INTO questions (question,answer) VALUES (?,?)", (question_text, answer_text))


    def get_command(self, question):
        if not question:
            sys.exit(apply_color_and_rest(Fore.RED,
                                          'Empty query. Give a query to get the command. Eg: termy --gpt "How do I duplicate a folder" '))

        cache_answer = self.checkQ(question)
        if cache_answer:
            return cache_answer

        prompt = "I am a command line translation tool for " + self.platform + "."
        prompt = prompt + """
        Ask me what you want to do and I will tell you how to do it in a unix command.
        Q: How do I copy a file
        cp filename.txt destination_filename.txt
        Q: How do I duplicate a folder?
        cp -a source_folder/ destination_folder/
        Q: How do display a calendar?
        cal
        Q: how do I convert a .heic file to jpg?
        convert source.heic destination.jpg
        Q: navigate to my desktop
        cd ~/Desktop/
        Q: How do I shutdown the computer?
        sudo shutdown -h now
        """

        temp_question = question
        if not ("?" in question):
            temp_question = question + "?"

        prompt = prompt + "Q: " + temp_question + "\n"
        openai.api_key = self.API_KEY
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )

        result = response['choices'][0]['text']
        self.insertQ(question, result)
        self.closeDB()
        return result.strip()

    def load_api_key_from_config(self):
        if not os.path.exists(GPT3_CONFIG):
            return None
        with open(GPT3_CONFIG, 'r') as file:
            config = json.load(file)
        return config.get('GPT3_API_KEY', None)

    def get_platform(self):
        platform = sys.platform
        if platform == "darwin":
            platform = "Mac"
        elif platform == "win32":
            platform = "Windows"
        else:
            platform = "Linux"
        return platform

    def save_api_key_in_config(self):
        with open(GPT3_CONFIG, 'w') as f:
            json.dump({'GPT3_API_KEY' : self.API_KEY}, f)
