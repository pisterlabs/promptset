#!/usr/bin/env python3
import os
import openai
import pinecone
import time
import sys
from collections import deque
from typing import Dict, List
from dotenv import load_dotenv
import os

# Set Variables
load_dotenv()


OBJECTIVE = '''Find a new mathematics theorem and prove it.Use mathematic notation to wright your proof and hypothesis.
1/ Construct definitions from your mental model.
2/ Intuitions about potential theorem
3/ Sketch of a proof .
4/ Formalized the proof.
5/take a statetment and abstract it until it becomes false
6/We have a question that we cannot answer so we ask another related question
7/Create new tools to solve a problem
8/Manipulating the model in some abstract non formal way.
9/We are very good at finding bigger steps of a problem.
10/1 agent who make a move and 1 agent who evaluate the move.'''
expertslist = "someone who alwyas try to prove wrong, a visionary perfectionist that always try to do better, ellon musk who try to ship a product rapidly and a mathematics researcher with experience in the field"
YOUR_FIRST_TASK = "Build a task list"
OPENAI_API_MODEL = "gpt-3.5-turbo"
#Print OBJECTIVE
print("\033[96m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(YOUR_FIRST_TASK)
print(OBJECTIVE)

# Configure OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create Pinecone index
table_name = YOUR_TABLE_NAME
dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
  pinecone.create_index(table_name,
                        dimension=dimension,
                        metric=metric,
                        pod_type=pod_type)

# Connect to the index
index = pinecone.Index(table_name)

# Task list
task_list = deque([])


def add_task(task: Dict):
  task_list.append(task)


def get_ada_embedding(text):
  if isinstance(text, str):
    text = text.replace("\n", " ")
  else:
    # handle the case where text is not a string, e.g. raise an exception or return None
    return None

  return openai.Embedding.create(
    input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


def openai_call(prompt: str,
                model: str = OPENAI_API_MODEL,
                temperature: float = 0.5,
                max_tokens: int = 100):
  if not model.startswith('gpt-'):
    # Use completion API
    response = openai.Completion.create(engine=model,
                                        prompt=prompt,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0)
    return response.choices[0].text.strip()
  else:
    # Use chat completion API
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      n=1,
      stop=None,
    )
    return response.choices[0].message.content.strip()


def task_creation_agent(corrections: str, objective: str, result: Dict,
                        task_description: str, task_list: List[str]):
  prompt = f"You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks you previously define: {', '.join(task_list)}. This is whate your professor and manager said about it,{corrections} .Based on the previous, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array."
  response = openai_call(prompt)
  new_tasks = response.split('\n')
  return [{"task_name": task_name} for task_name in new_tasks]


def prioritization_agent(this_task_id: int):
  global task_list
  task_names = [t["task_name"] for t in task_list]
  next_task_id = int(this_task_id) + 1
  prompt = f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
  response = openai_call(prompt)
  new_tasks = response.split('\n')
  task_list = deque()
  for task_string in new_tasks:
    task_parts = task_string.strip().split(".", 1)
    if len(task_parts) == 2:
      task_id = task_parts[0].strip()
      task_name = task_parts[1].strip()
      task_list.append({"task_id": task_id, "task_name": task_name})


def execution_agent(objective: str, task: str) -> str:
  context = context_agent(query=objective, n=5)
  #print("\n*******RELEVANT CONTEXT******\n")
  #print(context)
  prompt = f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
  return openai_call(prompt, temperature=0.7, max_tokens=2000)


def execution_agent2(thetask: str, objective: str) -> str:
  #print("\n*******RELEVANT CONTEXT******\n")
  #print(context)
  print(thetask)
  prompt = f"You are an AI who performs one task based on the following objective: {objective}.Execute the tasks in point 1,2,3,4 include in the following in 50 word max per task  #### {thetask}####  \nResponse:"
  res1 = {openai_call(prompt, temperature=0.7, max_tokens=2000)}
  print(res1)
  prompt1 = f'You are an AI who performs one task based on the following objective: {objective}.Execute the tasks in point 5,6,7 include in the following in 50 word max per task #### {thetask} #### \nResponse:'
  res2 = {openai_call(prompt1, temperature=0.7, max_tokens=2000)}
  print(res2)
  prompt3 = f'You are an AI who performs one task based on the following objective: {objective}.Execute the tasks in point 8,9,10 include in the following in 50 word max per task   #### {thetask} #### \nResponse:'
  res3 = {openai_call(prompt3, temperature=0.7, max_tokens=2000)}
  print(res3)
  return thetask, res1, res2, res3


def professor_agent(objective: str, this_task_id: int) -> str:
  context = context_agent(query=objective, n=5)
  #print("\n*******RELEVANT CONTEXT******\n")
  #print(context)
  prompt = f"You are a council composed of {expertslist} who verifiate tasks that an AI who try to accomplish {objective}.\nTake into account his previously completed tasks: {context}, are the next tasks he propose are coherent to solve {objective} given the max 2000token and his limited capabilities as an AI.Be extremly severe before answering True. Your answer will only following these rules, If you think tasks are coherent start your answer by True, if no return 5 detailed correction he can make"

  return openai_call(prompt, temperature=0.7, max_tokens=2000)


def context_agent(query: str, n: int):
  query_embedding = get_ada_embedding(query)
  results = index.query(query_embedding, top_k=n, include_metadata=True)
  #print("***** RESULTS *****")
  #print(results)
  sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
  return [(str(item.metadata['task'])) for item in sorted_results]


# Add the first task
first_task = {"task_id": 1, "task_name": YOUR_FIRST_TASK}

add_task(first_task)
# Main loop
task_id_counter = 1
while True:
  if task_list:
    # Print the task list
    print("\033[95m\033[1m" + "\n***** BabyAGI *****\n" + "\033[0m\033[0m")
    #for t in task_list:
    #print(str(t['task_id']) + ": " + t['task_name'])

    # Step 1: Pull the first task
    task = task_list.popleft()
    #print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
    print(str(task['task_id']) + ": " + task['task_name'])

    # Send to execution function to complete the task based on the context
    r1, r2, r3, r4 = execution_agent2(
      execution_agent(OBJECTIVE, task["task_name"]), task["task_name"])
    if isinstance(r1, set):
      r1 = "".join(str(item) for item in r1)
    if isinstance(r2, set):
      r2 = ", ".join(str(item) for item in r2)
    if isinstance(r3, set):
      r3 = "".join(str(item) for item in r3)
    if isinstance(r4, set):
      r4 = ", ".join(str(item) for item in r4)

# Concatenate strings
    result = r1 + r2 + r3 + r4
    this_task_id = int(task["task_id"])
    #print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
    print(result)

    # Step 2: Enrich result and store in Pinecone
    enriched_result = {
      'data': result
    }  # This is where you should enrich the result if needed
    result_id = f"result_{task['task_id']}"
    vector = enriched_result[
      'data']  # extract the actual result from the dictionary
    index.upsert([(result_id, get_ada_embedding(vector), {
      "task": task['task_name'],
      "result": result
    })])

  # Step 3: Create new tasks and reprioritize task list
  test = False
  while test == False:
    new_tasks = task_creation_agent(result, OBJECTIVE, enriched_result,
                                    task["task_name"],
                                    [t["task_name"] for t in task_list])
    result = professor_agent(OBJECTIVE, [t["task_name"] for t in task_list])
    test = result[:4] == "True"
    if test == False:
      print("Iterating...")
      task_list = new_tasks

  for new_task in new_tasks:
    task_id_counter += 1
    new_task.update({"task_id": task_id_counter})
    add_task(new_task)
  prioritization_agent(this_task_id)

  time.sleep(1)  # Sleep before checking the task list again
