"""
Actions: contains all the functions used to compose the low-high level 
actions for any agent programs that use these functions.
"""

import os
import openai
from typing import List, Tuple
from collections import deque
import pinecone

# pincone setup
USE_PINECONE = False  # flag to set pincone usage on or off (default: False)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_TABLE = os.getenv("PINECONE_TABLE")

# Create Pinecone index
DIMENSION = 1536
METRIC = "cosine"
POD_TYPE = "p1"
if USE_PINECONE and PINECONE_TABLE not in pinecone.list_indexes():
    pinecone.create_index(
        PINECONE_TABLE, dimension=DIMENSION, metric=METRIC, pod_type=POD_TYPE
    )

# Init Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


task_creation_template = """
You are a task creation AI that uses the result of an execution agent to 
create new tasks with the following objective: {objective}. The last 
completed task has the result: {result}. This result was based on this
task description: {task_description}. These are incomplete tasks: 
{incomplete_tasks}. Based on the result, create new tasks to be completed 
by the AI system that do not overlap with incomplete tasks. Return the 
result as a list that is not numbered and simply lists each task in its
own line, like:

First task
Second task
Third task

Do not include anything else except the list in your response.
"""
task_prioritization_template = """
You are a task prioritization AI tasked with cleaning the formatting of
and reprioritizing the following tasks: {task_names}. Consider the 
ultimate objective of your team: {objective}. 
Do not remove any tasks. Return the result as a numbered list, like:

#. First task
#. Second task

Start the task list with number 1 and do not include anything else 
except the list in your response.
"""
task_execution_template = """
You are an AI that performs one task based on the following objective: 
{objective}.\nTake into account these previously completed tasks: 
{context}...and your assigned task: {task}\n. What is your response? 
Make sure to respond with a detailed solution to the assigned task you 
have been given only, and do not address any other tasks or make any lists. 
your response should be in paragraph form.
"""
task_stop_or_not_template = """
You are an AI that assess task completion for the following objective: 
{objective}. Take into account these previously completed tasks: {context}.
Has the objective been achieved? Answer with only yes or no. Only answer 
with yes if you think this is the best answer possible.
"""


def task_execution_prompt_builder(globals_: dict) -> str:
    """
    This function builds and returns the execution prompt for GPT to take
    in as input when executing a task. It also prints the next task in
    the task list.

    Args:
        globals_ (dict): The globals dictionary

    Returns:
        str: The prompt for GPT task execution
    """
    task_list = globals_["task_list"]
    task = task_list.popleft()
    globals_["current_task"] = task

    print("\033[92m\033[1m" + "\n***** NEXT TASK *****\n" + "\033[0m\033[0m")
    print(str(task["id"]) + ": " + task["name"])

    context = get_context(globals_)
    return task_execution_template.format(
        objective=globals_["objective"], task=globals_["current_task"], context=context
    )


def task_execution_handler(response: str, globals_: dict) -> None:
    """
    This function handles the GPT response corresponding to the last task
    execution, allows for the result to be further enriched and prints the
    resultant GPT response from executing the task. It also prints the result

    Args:
        response (str): The GPT response from task execution
        globals_ (dict): The globals dictionary
    """
    enriched_result = {
        "data": response
    }  # This is where you should enrich the result if needed
    globals_["result"] = enriched_result

    """Use Pinecone, not currently setup"""
    id_ = globals_["current_task"]["id"]
    result_id = f"result_{id_}"
    vector = enriched_result["data"]  # extract the actual result from the dictionary
    if USE_PINECONE:
        index = pinecone.Index(index_name=PINECONE_TABLE)
        index.upsert(
            [
                (
                    result_id,
                    get_ada_embedding(vector),
                    {"task": globals_["current_task"]["name"], "result": response},
                )
            ]
        )

    print("\033[93m\033[1m" + "\n***** TASK RESULT *****\n" + "\033[0m\033[0m")
    print(globals_["result"]["data"])

    return "done"


def task_creation_prompt_builder(globals_: dict) -> str:
    """
    This function builds and returns the prompt for GPT task
    creation so GPT can create task lists.

    Args:
        globals_ (dict): The globals dictionary

    Returns:
        str: The prompt for GPT task creation
    """
    incomplete_tasks = [t["name"] for t in globals_["task_list"]]
    return task_creation_template.format(
        objective=globals_["objective"],
        result=globals_["result"],
        task_description=globals_["current_task"].get("name", "default"),
        incomplete_tasks=incomplete_tasks,
    )


def task_creation_handler(response: str, globals_: dict):
    """
    This function handles the GPT response corresponding to the task
    creation prompt built by the task creation prompt builder and
    prints the resultant GPT response that is creating new tasks.

    Args:
        response (str): The GPT response from task creation
        globals_ (dict): The globals dictionary

    Returns:
        str: The status of the task creation handler

    """
    new_tasks = response.split("\n")
    if len(globals_["task_list"]) > 0:
        id_ = globals_["task_list"][-1]["id"] + 1
    else:
        id_ = 1
    task_list = [
        {"id": id_ + i, "name": task_name} for i, task_name in enumerate(new_tasks)
    ]
    globals_["task_list"] = deque(task_list)

    print("\033[89m\033[1m" + "\nTASK LIST:" + "\033[0m\033[0m")
    for t in task_list:
        print(t["name"])

    return "done"


def task_prioritization_prompt_builder(globals_: dict) -> str:
    """
    This function builds and returns the prompt for GPT task prioritization
    so existing tasks can be re-prioritized.

    Args:
        globals_ (dict): The globals dictionary

    Returns:
        str: The prompt for GPT task prioritization
    """
    task_list = globals_["task_list"]
    current_task = globals_["current_task"]
    task_names = [t["name"] for t in task_list]
    current_task_id = int(current_task["id"]) + 1
    objective = globals_["objective"]
    return task_prioritization_template.format(
        task_names=task_names, objective=objective, starting_id=current_task_id
    )


def task_prioritization_handler(response: str, globals_: dict):
    """
    This function handles the GPT response corresponding to the task
    prioritization prompt built by the task prioritization prompt builder and
    prints the resultant GPT response that is re-prioritizing existing tasks.
    """
    new_tasks = response.split("\n")
    task_list = deque([])
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = int(task_parts[0].strip())
            task_name = task_parts[1].strip()
            task_list.append({"id": task_id, "name": task_name})
    globals_["task_list"] = task_list
    globals_["current_task"] = {}
    print("\033[94m\033[1m" + "\n***** RE-PRIORITIZED LIST *****\n" + "\033[0m\033[0m")
    for t in task_list:
        print(str(t["id"]) + ": " + t["name"])
    return "done"


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def get_context(globals_: dict) -> List[Tuple[str]]:
    """
    Get the current context (task list) from the dictionary state variable, globals_
    """
    """Use Pinecone, not currently setup"""
    if USE_PINECONE:
        query = globals_["objective"]
        query_embedding = get_ada_embedding(query)
        index = pinecone.Index(index_name=PINECONE_TABLE)
        results = index.query(query_embedding, top_k=5, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]
    return globals_["task_list"]


def task_stop_or_not_prompt_builder(globals_: dict) -> str:
    """
    This function builds and returns the task stop or not prompt for GPT
    in order to reason about the objective completeness when the user
    stops the agent loop.

    Args:
        globals_ (dict): The globals dictionary

    Returns:
        str: The prompt for GPT task stop or not
    """
    context = get_context(globals_)
    return task_stop_or_not_template.format(
        objective=globals_["objective"], context=context
    )


def task_stop_or_not_handler(response: str, globals_: dict) -> None:
    """
    This function handles the GPT response corresponding to the task stop
    or not prompt built by the task stop or not prompt builder and prints
    the resultant GPT response that is reasoning about the objective
    completeness when the user stops the agent loop.
    """
    globals_["keep_going"] = response.strip().lower() != "yes"
    print("\033[94m\033[1m" + "\n*****TASK CONTINUATION*****\n" + "\033[0m\033[0m")
    print(globals_["keep_going"])
    return "done" if globals_["keep_going"] else "stop"
