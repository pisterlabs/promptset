import json
from typing import Annotated, List, Union

import database
import openai
import user_auth_api
from fastapi import HTTPException, Response, status
from pydantic import BaseModel

with open('./config.json') as f:    
    config = json.load(f)


class UserRequestedTasks(BaseModel):
    requested_tasks: List[str]

class SubTask(BaseModel):
    sub_task_id: int
    parent_task_id: int
    sub_task_name: str
    is_completed: bool = False
    task_time_estimate_in_minutes: str
    task_priority: str

class Task(BaseModel):
    task_id: int
    user_id: int
    task_name: str
    is_accepted: bool = True
    sub_tasks: List[SubTask] = []

class TimeSlot(BaseModel):
    slot_id: int
    user_id: int
    slot_time: str
    sub_tasks: List[SubTask] =[]


async def _put_sub_tasks_in_db(parent_task_id:int, all_sub_tasks_str:dict) -> List[SubTask]:
    conn = await database.get_db_connection()
    cursor = conn.cursor() 
    
    sub_tasks = []

    for sub_task_str in all_sub_tasks_str:
        
        task_name = sub_task_str['task_detail_name']
        task_time_estimate = sub_task_str['task_time_estimate']
        task_priority = sub_task_str['task_priority']
        
        cursor.execute('INSERT INTO sub_tasks (parent_task_id, sub_task_name, task_time_estimate_in_minutes, task_priority) VALUES (?, ?, ?, ?)', (parent_task_id, task_name, task_time_estimate, task_priority))
        conn.commit()

        sub_task = SubTask(
            sub_task_id=cursor.lastrowid,
            parent_task_id=parent_task_id,
            sub_task_name=task_name,
            task_time_estimate_in_minutes=task_time_estimate,
            task_priority=task_priority
        )

        sub_tasks.append(sub_task)
    
    return sub_tasks


async def _put_tasks_in_db(all_tasks_str:dict,current_user:user_auth_api.User) -> List[Task]:
    conn = await database.get_db_connection()
    cursor = conn.cursor() 
    
    tasks = []

    for task_str in all_tasks_str:

        task_name = task_str['task_name']
        user_id = int(current_user.id)

        cursor.execute('INSERT INTO task (user_id, task_name, is_accepted) VALUES (?, ?, ?)', (user_id, task_name, True))
        conn.commit()

        task = Task(
            task_id=cursor.lastrowid,
            task_name=task_name,
            is_accepted=True,
            user_id=user_id
        )

        # sub task parsing ====================

        all_sub_tasks_str = task_str['task_details']

        sub_tasks = await _put_sub_tasks_in_db(
            all_sub_tasks_str=all_sub_tasks_str,
            parent_task_id=task.task_id,
        )

        task.sub_tasks = sub_tasks

        tasks.append(task)

    return tasks


async def _put_slots_in_db(all_slots_str:dict,current_user:user_auth_api.User) -> List[TimeSlot]:
    conn = await database.get_db_connection()
    cursor = conn.cursor() 
    
    tasks = []

    for slot_str in all_slots_str:

        slot_time = slot_str['slot_time']
        user_id = int(current_user.id)

        cursor.execute('INSERT INTO slot (user_id, slot_time) VALUES (?, ?)', (user_id, slot_time))
        conn.commit()

        task = TimeSlot(
            task_id=cursor.lastrowid,
            task_name=slot_time,
            user_id=user_id
        )

        # sub task parsing ====================

        all_sub_tasks_str = slot_str['task_details']

        sub_tasks = await _put_sub_tasks_in_db(
            all_sub_tasks_str=all_sub_tasks_str,
            parent_task_id=task.task_id,
        )

        task.sub_tasks = sub_tasks

        tasks.append(task)

    return tasks


async def sub_tasks(tasks:UserRequestedTasks,response: Response,current_user: user_auth_api.User):
    requested_tasks_as_string = _prepare_input_tasks(tasks)

    try:
        
        generated_tasks = await _get_generated_tasks_from_openai(requested_tasks_as_string)

        parsed_tasks = _parse_generated_tasks(generated_tasks)

        tasks = await _put_tasks_in_db(parsed_tasks,current_user)

        return tasks

    except openai.error.Timeout as e:
        #Handle timeout error, e.g. retry or log
        response.status_code = status.HTTP_408_REQUEST_TIMEOUT
        return {"details":f"OpenAI API request timed out: {e}"}

    except openai.error.APIError as e:
        #Handle API error, e.g. retry or log
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"details":f"OpenAI API returned an API Error: {e}"}
        
    except openai.error.APIConnectionError as e:
        #Handle connection error, e.g. check network or log
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"details":f"OpenAI API request failed to connect: {e}"}
        
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"details":f"OpenAI API request was invalid: {e}"}
        
    except openai.error.AuthenticationError as e:
        #Handle authentication error, e.g. check credentials or log
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"details":f"OpenAI API request was not authorized: {e}"}
        
    except openai.error.PermissionError as e:
        #Handle permission error, e.g. check scope or log
        response.status_code = status.HTTP_403_FORBIDDEN
        return {"details":f"OpenAI API request was not permitted: {e}"}
        
    except openai.error.RateLimitError as e:
        #Handle rate limit error, e.g. wait or log
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"details":f"OpenAI API request exceeded rate limit: {e}"}

async def create_tasks(tasks:UserRequestedTasks,response: Response,current_user: user_auth_api.User):
    
    requested_tasks_as_string = _prepare_input_tasks(tasks)

    try:
        
        # generated_tasks = await _get_generated_tasks_from_openai(requested_tasks_as_string)
        # scheduled_tasks = await _get_scheduled_tasks_from_openai(generated_tasks)

        global _scheduled_tasks
        parsed_tasks = _parse_scheduled_tasks(_scheduled_tasks)
        # parsed_tasks = _parse_scheduled_tasks(scheduled_tasks)

        slots = _put_slots_in_db(parsed_tasks,current_user)

        return slots

    except openai.error.Timeout as e:
        #Handle timeout error, e.g. retry or log
        response.status_code = status.HTTP_408_REQUEST_TIMEOUT
        return {"details":f"OpenAI API request timed out: {e}"}

    except openai.error.APIError as e:
        #Handle API error, e.g. retry or log
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"details":f"OpenAI API returned an API Error: {e}"}
        
    except openai.error.APIConnectionError as e:
        #Handle connection error, e.g. check network or log
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"details":f"OpenAI API request failed to connect: {e}"}
        
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"details":f"OpenAI API request was invalid: {e}"}
        
    except openai.error.AuthenticationError as e:
        #Handle authentication error, e.g. check credentials or log
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"details":f"OpenAI API request was not authorized: {e}"}
        
    except openai.error.PermissionError as e:
        #Handle permission error, e.g. check scope or log
        response.status_code = status.HTTP_403_FORBIDDEN
        return {"details":f"OpenAI API request was not permitted: {e}"}
        
    except openai.error.RateLimitError as e:
        #Handle rate limit error, e.g. wait or log
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"details":f"OpenAI API request exceeded rate limit: {e}"}
        

async def _get_generated_tasks_from_openai(tasks:str):
    prompts = config["prompts"]

    response = openai.ChatCompletion.create(
        model=config["models"]["gpt-3.5"]["model-name"],
        messages=[
            {"role": "system", "content": prompts["task_division_system_prompt"]},
            {
                "role": "user",
                "content": prompts["task_division_prompt_prefix"]
                + tasks
                + prompts["task_division_prompt_suffix"],
            },
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n\n"],
    )

    openai_response = response["choices"][0]["message"]["content"]

    return openai_response

async def _get_scheduled_tasks_from_openai(generated_tasks:str):

    prompts = config["prompts"]

    response = openai.ChatCompletion.create(
        model=config["models"]["gpt-3.5"]["model-name"],
        messages=[
            {"role": "system", "content": prompts['day_schedule_system_prompt']},
            {
                "role": "user",
                "content": prompts['day_schedule_prompt_prefix']
                + generated_tasks
                + prompts['day_schedule_system_prompt_suffix'],
            },
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n\n"],
    )

    openai_response = response["choices"][0]["message"]["content"]

    return openai_response


def _prepare_input_tasks(tasks:UserRequestedTasks):
    
    requested_tasks = tasks.requested_tasks
    task_list=''

    for index, task in enumerate(requested_tasks):
        numbered_task = f'\n{index}. {task}'
        task_list = task_list + numbered_task

    return task_list


def _parse_generated_tasks(tasks_output: str) -> dict:
    tasks = []

    for task in tasks_output.split("Task: "):
        current_task = {}
        if task.strip() == "":
            continue
        task_name, task_details = task.split("\nMini Atomic Task | Time Estimate | Priority Level |")
        task_name = task_name.strip()
        task_details = task_details.strip()
        current_task["task_name"] = task_name
        current_task["task_details"] = []
        for task_detail in task_details.split("\n"):
            current_task_detail = {}
            if task_detail.strip() == "" or len(task_detail.split("|")) != 4:
                continue
            task_detail = task_detail.strip()
            task_detail_name, task_time_estimate, task_priority, _ = task_detail.split("|")
            current_task_detail["task_detail_name"] = task_detail_name.strip()
            current_task_detail["task_time_estimate"] = task_time_estimate.strip()
            current_task_detail["task_priority"] = task_priority.strip()
            current_task["task_details"].append(current_task_detail)
        tasks.append(current_task)
    return tasks

def _parse_scheduled_tasks(tasks_output: str) -> List[TimeSlot]:

    available_slots = []

    for slot_data in tasks_output.split("Time Slot: "):
        
        if slot_data.strip() == "":
            continue
        
        slot_and_sub_tasks = slot_data.split("\n")

        slot_time = slot_and_sub_tasks[0]
        sub_tasks_str = slot_and_sub_tasks[1:]

        sub_tasks = []

        for current_sub_task in sub_tasks_str:

            if current_sub_task.strip() == "" or len(current_sub_task.split("|")) != 4:
                continue

            current_sub_task = current_sub_task.strip()
            sub_task_name, sub_task_time_estimate, sub_task_priority,_ = current_sub_task.split("|")

            sub_task = SubTask(
                sub_task_name=sub_task_name.strip(),
                task_time_estimate_in_minutes=sub_task_time_estimate.strip(),
                task_priority=sub_task_priority.strip(),
            )

            sub_tasks.append(sub_task)


        current_slot = TimeSlot(
            slot_time=slot_time,
            sub_tasks=sub_tasks
        )

        available_slots.append(current_slot)

    return available_slots



# Get SubTask by ID
async def get_subtask(sub_task_id: int) -> SubTask:
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sub_tasks WHERE sub_task_id = ?", (sub_task_id,))
    row = cursor.fetchone()
    if row is not None:
        sub_task = SubTask(
            sub_task_id=row[0],
            parent_task_id=row[1],
            sub_task_name=row[2],
            task_time_estimate_in_minutes=row[3],
            task_priority=row[4],
            is_completed=bool(row[5]),
        )
        return sub_task
    return None

# Update SubTask by ID
async def update_subtask_in_db(sub_task_id: int, sub_task: SubTask):
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sub_tasks SET parent_task_id = ?, sub_task_name = ?, is_completed = ?, task_time_estimate_in_minutes = ?, task_priority = ? WHERE sub_task_id = ?",
        (
            sub_task.parent_task_id,
            sub_task.sub_task_name,
            sub_task.is_completed,
            sub_task.task_time_estimate_in_minutes,
            sub_task.task_priority,
            sub_task_id,
        ),
    )
    conn.commit()

# Delete SubTask by ID
async def delete_subtask_from_db(sub_task_id: int):
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sub_tasks WHERE sub_task_id = ?", (sub_task_id,))
    conn.commit()

# Get Task by ID
async def get_task(task_id: int) -> Task:
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM task WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()
    if row is not None:
        sub_tasks = await get_subtasks_for_task(task_id)
        task = Task(
            task_id=row[0],
            user_id=row[1],
            task_name=row[2],
            is_accepted=row[3],
            sub_tasks=sub_tasks,
        )
        return task
    return None

# Get SubTasks for Task by ID
async def get_subtasks_for_task(task_id: int) -> List[SubTask]:
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sub_tasks WHERE parent_task_id = ?", (task_id,))
    rows = cursor.fetchall()
    sub_tasks = []

    for row in rows:

        sub_task = SubTask(
            sub_task_id=row[0],
            parent_task_id=row[1],
            sub_task_name=row[2],
            task_time_estimate_in_minutes=row[3],
            task_priority=row[4],
            is_completed=bool(row[5]),
        )

        sub_tasks.append(sub_task)
    return sub_tasks

# Update Task by ID
async def update_task_in_db(task_id: int, task: Task):
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE task SET user_id = ?, task_name = ?, is_accepted = ? WHERE task_id = ?",
        (task.user_id, task.task_name, task.is_accepted, task_id),
    )
    conn.commit()
    for sub_task in task.sub_tasks:
        await update_subtask_in_db(sub_task.sub_task_id, sub_task)

# Delete Task by ID
async def delete_task_from_db(task_id: int):
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM task WHERE task_id = ?", (task_id,))
    conn.commit()

# Get task list for user
async def get_tasks_for_user(current_user:user_auth_api.User) -> List[Task]:
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM task WHERE user_id = ?", (current_user.id,))
    rows = cursor.fetchall()
    tasks = []
    for row in rows:
        sub_tasks = await get_subtasks_for_task(row[0])
        task = Task(
            task_id=row[0],
            user_id=row[1],
            task_name=row[2],
            is_accepted=row[3],
            sub_tasks=sub_tasks,
        )
        tasks.append(task)
    return tasks



async def get_subtasks_for_parent(parent_task_id: int) -> List[SubTask]:
    conn = await database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sub_tasks WHERE parent_task_id = ?", (parent_task_id,))
    rows = cursor.fetchall()
    sub_tasks = []
    for row in rows:
        sub_task = SubTask(
            sub_task_id=row[0],
            parent_task_id=row[1],
            sub_task_name=row[2],
            task_time_estimate_in_minutes=row[3],
            task_priority=row[4],
            is_completed=bool(row[5]),
        )
        sub_tasks.append(sub_task)
    return sub_tasks

_generated_tasks = """
Task: Wash utensils
Mini Atomic Task | Time Estimate | Priority Level |
Wash dishes | 10 | High |
Dry dishes and put them away | 5 | Medium |

Task: Water plants
Mini Atomic Task | Time Estimate | Priority Level |
Gather watering can and fill with water | 5 | Low |
Water plants | 10 | High |
Clean up any spills | 5 | Low |

Task: Talk to family
Mini Atomic Task | Time Estimate | Priority Level |
Decide on topic of conversation | 5 | Low |
Initiate conversation | 5 | High |
Listen actively and respond appropriately | 30 | High |

Task: Go to bed timely
Mini Atomic Task | Time Estimate | Priority Level |
Finish any pending work | 30 | High |
Put away any distractions | 5 | Medium |
Brush teeth and change into comfortable clothes | 10 | High |
Set alarm for next day | 5 | Medium |
Get into bed and relax | 20 | High |
Sleep | 360 | High |

Note: Time estimates and priority levels can vary based on individual needs and preferences. It is important to work with the person to create a personalized plan that works best for them.
"""

_scheduled_tasks = """
Time Slot: 8:00 AM - 9:00 AM
Check bank balance | 5 | High |

Time Slot: 9:00 AM - 10:00 AM
Fill watering can | 2 | Medium |
Water plants | 10 | High |
Clean up spilled water | 3 | Low |

Time Slot: 10:00 AM - 11:00 AM
Decide who to call | 5 | Medium |
Make phone call | 15 | High |
Listen to family member | 20 | High |
Take notes if necessary | 5 | Medium |

Time Slot: 11:00 AM - 12:00 PM
Withdraw cash | 10 | High |
Go to broker's office | 15 | High |
Pay rent | 5 | High |

Time Slot: 12:00 PM - 1:00 PM
Lunch Break

Time Slot: 1:00 PM - 2:00 PM
Set alarm | 2 | High |
Brush teeth | 5 | High |
Change into pajamas | 2 | Medium |

Time Slot: 2:00 PM - 3:00 PM
Read for 20 minutes | 20 | Medium |

Time Slot: 3:00 PM - 4:00 PM
Break

Time Slot: 4:00 PM - 5:00 PM
Check bank balance | 5 | High |

Time Slot: 5:00 PM - 6:00 PM
Fill watering can | 2 | Medium |
Water plants | 10 | High |
Clean up spilled water | 3 | Low |

Time Slot: 6:00 PM - 7:00 PM
Break

Time Slot: 7:00 PM - 8:00 PM
Decide who to call | 5 | Medium |
Make phone call | 15 | High |
Listen to family member | 20 | High |
Take notes if necessary | 5 | Medium |

Time Slot: 8:00 PM - 9:00 PM
Read for 20 minutes | 20 | Medium |

Time Slot: 9:00 PM - 10:00 PM
Turn off lights | 2 | High |

Note: The time slots can be adjusted as per the individual's preference and availability."""
