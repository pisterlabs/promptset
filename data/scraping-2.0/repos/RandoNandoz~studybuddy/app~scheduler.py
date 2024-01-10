"""
This file is used to schedule the tasks given.
"""

import openai
import dotenv
import os
import json

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def schedule(tasks: list):
    """
    Given a list of tasks (tasks are represented as dictionaries), schedule the tasks and return the schedule as a dictionary with description and start and end times.
    
    SAMPLE DATA:
    ```
    {
    "description": "Work on project",
    "time_required": 120,
    "priority": "highest",
    "diffculty": "high",
    "fun": True
    },
    {
        "description": "Clean the house",
        "time_required": 60,
        "priority": "high",
        "diffculty": "average",
        "fun": False
    },
    {
        "description": "Go to the gym",
        "time_required": 80,
        "priority": "average",
        "diffculty": "lowest",
        "fun": True
    }
    ```
    """
    SYSTEM_INSTRUCTIONS = "Given a list of tasks in Python dictionary format, output a schedule for the day in raw JSON format with no escaping, all on one line, with the descriptions and times in the day to do them. Try to balance priority and difficulty as to complete high priority tasks, but not burn out the user. Consider all your data. DO NOT OUTPUT ANYTHING ELSE"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTIONS
            },
            {
                "role": "user",
                "content": str(tasks)
            }
        ]
    )
    
    return json.loads(completion.choices[0].message.content) # type: ignore # completion.choices guaranteed to be non-empty, pylance is just dumb, again
