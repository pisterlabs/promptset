import openai
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()
import logging
from pathlib import Path

class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages.
    """
    BLACK = '\033[0;30m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[0;37m'
    RESET = '\033[0m'

    LEVEL_COLORS = {
        'DEBUG': CYAN,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': PURPLE,
    }

    def format(self, record):
        log_level_color = self.LEVEL_COLORS.get(record.levelname, self.WHITE)
        log_level_name = f'{log_level_color}{record.levelname}{self.RESET}'
        record.levelname = log_level_name
        return super().format(record)

class GPTTeacher(object):
    def __init__(self, model="gpt-3.5-turbo-0613"):
        self.model = model
        # connect to openai api
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.messages_system = []
        self.messages_task = []

        # logging
        logs_directory = Path('logs')
        if not logs_directory.exists():
            logs_directory.mkdir()
        self.logging = logging.getLogger('gpt_teacher')
        self.logging.setLevel(logging.DEBUG)
        if not self.logging.hasHandlers():
            file_handler = logging.FileHandler('logs/gpt_teacher.log', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logging.addHandler(file_handler)

        # Stream handler
        # stream_handler = logging.StreamHandler(sys.stdout)
        # stream_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        # self.logging.addHandler(stream_handler)

    def append_system_message(self, message):
        self.messages_system.append({"role": "system", "content": message})
        self.logging.info(f"System Messages: {self.messages_system}")
    
    def get_score(self, task_id, task_description, container, shell):
        self.logging.info(f"Start grading Task {task_id}...")
        self.messages_task = []
        self.messages_task.append({"role": "user", "content": f"The student is doing this task:\n{task_description}\n------\nNow the student has submitted the solution. Please check the submission using command 'command'. You can use command 'grade' to give a score."})
        observation = ""
        reward = 10
        comments = "Good job!"

        for i in range(10):
            self.messages_task.append({"role": "user", "content": f"{observation}\nWhat's your next action? (respond in JSON format)"})
            self.logging.info(f"Task Messages: {self.messages_task}")
            chat_completion = openai.ChatCompletion.create(model=self.model,
                                                        messages=(self.messages_system+self.messages_task))
            action = chat_completion["choices"][0]["message"]["content"] # type: ignore
            self.logging.info(f"Chat completion: {chat_completion}")
            self.messages_task.append({"role": "assistant", "content": action})
            self.logging.info(f"Action: {action}")
            try:
                action_obj = json.loads(action)
                if action_obj['action'] == "grade":
                    comments = action_obj['comments']
                    reward = action_obj['score']
                    break
                exec_result = container.exec_run([shell, '-c', action_obj['command']]) # type: ignore
                observation = {"obs": exec_result.output.decode('utf-8')}
            except:
                observation = {"obs": "Invalid action, respond in JSON format."}
            self.logging.info(f"Observation: {observation}")

        return reward, comments