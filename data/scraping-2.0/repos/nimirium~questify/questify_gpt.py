import json
from typing import List, Optional

import openai
from dotenv import load_dotenv
import os

from models.task import Task
from models.to_do_context import ToDoContext

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def questify_tasks(tasks: List[Task], context: Optional[ToDoContext] = None) -> dict:
    to_do = [t.model_dump() for t in tasks]
    prompt = "I have a list of tasks that I need to do. Turn each of them into a quest with a small storyline.\n\n" \
             + json.dumps(to_do) \
             + ' \n\n reply in a json format like \n' \
               '{"questlineName": "...", "quests": {"[id]": {"originalTask": "...", "questName": "...", "questDescription": "..."}}}' \
               'where [id] is the id of the task.\n\n'
    if context:
        prompt += f"By the way, the to-do list title is '{context.title}' - make the questlineName sound similar but quest-like. " \
                  f"You may also use the information from the title in the quests themselves." \
                  f"The current time is {context.time}. " \
                  f"Do not include the exact time in your response, instead say morning / noon / afternoon / night, etc.\n"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{
            "role": "user",
            "content": prompt,
        }]
    )
    return json.loads(completion["choices"][0]["message"]["content"])
