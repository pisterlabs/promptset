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
import firebase_admin
from firebase_admin import credentials, firestore
from associations import Association

#Set Variables
load_dotenv()

# Set up Firebase
PATH_TO_FIREBASE_CONFIG= os.getenv("PATH_TO_FIREBASE_CONFIG", "")
cred = credentials.Certificate(PATH_TO_FIREBASE_CONFIG)
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection("associations")

# Set API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

# Use GPT-3 model
USE_GPT4 = False
if USE_GPT4:
    print("\033[91m\033[1m"+"\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"+"\033[0m\033[0m")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
assert PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
assert PINECONE_ENVIRONMENT, "PINECONE_ENVIRONMENT environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Project config
OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OBJECTIVE", "")
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"

YOUR_FIRST_TASK = os.getenv("FIRST_TASK", "")
assert YOUR_FIRST_TASK, "FIRST_TASK environment variable is missing from .env"

#Print OBJECTIVE
print("\033[96m\033[1m"+"\n*****OBJECTIVE*****\n"+"\033[0m\033[0m")
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
    pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

# Connect to the index
index = pinecone.Index(table_name)

# Task list
task_list = deque([])

def add_task(task: Dict):
    task_list.append(task)

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

def openai_call(prompt: str, use_gpt4: bool = False, temperature: float = 0.5, max_tokens: int = 100):
    if not use_gpt4:
        #Call GPT-3 DaVinci model
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()
    else:
        #Call GPT-4 chat model
        messages=[{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()

def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str], gpt_version: str = 'gpt-3'):
    prompt = f"You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array."
    response = openai_call(prompt, USE_GPT4)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks]

def prioritization_agent(this_task_id:int, gpt_version: str = 'gpt-3'):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id)+1
    prompt = f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt, USE_GPT4)
    new_tasks = response.split('\n')
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

def execution_agent(objective:str,task: str, gpt_version: str = 'gpt-3') -> str:
    #context = context_agent(index="quickstart", query="my_search_query", n=5)
    context=context_agent(index=YOUR_TABLE_NAME, query=objective, n=5)
    #print("\n*******RELEVANT CONTEXT******\n")
    #print(context)
    prompt =f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
    return openai_call(prompt, USE_GPT4, 0.7, 2000)
"""
def context_agent(query: str, index: str, n: int):
    query_embedding = get_ada_embedding(query)
    index = pinecone.Index(index_name=index)
    results = index.query(query_embedding, top_k=n, include_values=True,
    include_metadata=True)
    #print("***** RESULTS *****")
    #print(results)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)  
    print(sorted_results)  
    #exit()
    return [(str(item.metadata['task'])) for item in sorted_results]
"""
def context_agent(query: str, index: str, n: int):
    query_embedding = get_ada_embedding(query)
    index = pinecone.Index(index_name=index)
    results = index.query(query_embedding, top_k=n, include_values=True, include_metadata=True)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    context = [(str(item.metadata["task"])) for item in sorted_results]
    vectors = [item.values for item in sorted_results]

    # Retrieve associated information from the Firebase database
    associated_info = []
    associations = db.collection(u'associations').stream()
    for association in associations:
        if any(vector in vectors for vector in [association.to_dict()['vector1'], association.to_dict()['vector2']]):
            associated_info.append(association.to_dict()['association_description'])
            print("Associated info: ")

    return context + associated_info


def find_new_associations(table_name, new_vector, subject, text):
    # We take in the subject and text of the new vector
    # We get chat gpt to look for 3 new topics
    # We query pinecone for the 3 new topics
    # We check with chat gpt if the new topics are related to the subject
    # If the are, we ask chat gpt to give us a description of the connection
    # Save the new association


    index = pinecone.Index(index_name=table_name)

    # Initialize the list of new associations
    new_associations = []

    # Set how many possible associations per topic we want to find
    num_associations = 2

    # Get the new topics
    new_topics = get_possible_related_topics(subject)

    for topic in new_topics:
        if topic == "":
            continue
        # Get the vector for the topic
        print(topic)
        topic_vector_query = get_ada_embedding(topic)

        results = index.query(topic_vector_query, top_k=num_associations, include_values=True, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)

        contexts = [(str(item.metadata["task"])) for item in sorted_results]
        vectors = [item.values for item in sorted_results]

        for context, vector in zip(contexts, vectors):
            # Use OpenAI API to look for connections between the new information and the existing knowledge
            prompt_check = f"""
            You are an AI with associative memory and cross-disciplinary thinking capabilities.
            Given the new information: {text}
            And the existing knowledge: {context}
            Identify if there any connections or associations between the new information and the existing knowledge.
            RESPOND WITH: Yes or No
            """
            response = openai_call(prompt_check, temperature=0.7, max_tokens=100)
            if response == "Yes":
                # Use OpenAI API to describe the connection
                prompt_describe = f"""
                You are an AI with associative memory and cross-disciplinary thinking capabilities.
                Given the new information: {text}
                And the existing knowledge: {context}
                Identify and describe any connections or associations between the new information and the existing knowledge.
                """
                response_2 = openai_call(prompt_describe, temperature=0.7, max_tokens=400)
                new_associations.append(Association(new_vector, vector, response_2))
            else:
                continue

    return new_associations


def get_possible_related_topics(prompt: str, n: int = 3):
    # Construct the full prompt to ask for cross-disciplinary connections
    full_prompt = f"Think across disciplines and suggest {n} topics that are not directly related to the following knowledge, but may have valuable information or insights: {prompt}"
    
    # Call GPT API
    role = "You are an AI with associative memory and cross-disciplinary thinking capabilities."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": role,
         "role": "user", "content": full_prompt}
        ]
    )
    # Parse the response to extract the suggestions
    suggestions = completion.choices[0].message['content'].strip().split('\n')[:n]
    
    return suggestions






















# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}

add_task(first_task)
# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m"+"\n*****TASK LIST*****\n"+"\033[0m\033[0m")
        for t in task_list:
            print(str(t['task_id'])+": "+t['task_name'])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m"+"\n*****NEXT TASK*****\n"+"\033[0m\033[0m")
        print(str(task['task_id'])+": "+task['task_name'])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE,task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m"+"\n*****TASK RESULT*****\n"+"\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in Pinecone
        enriched_result = {'data': result}  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = enriched_result['data']  # extract the actual result from the dictionary
        embedding = get_ada_embedding(vector)
        index.upsert([(result_id, embedding,{"task":task['task_name'],"result":result})])

        # Step 2.5: Use associative memory agent to find new associations
        new_associations = find_new_associations(YOUR_TABLE_NAME, embedding, task["task_name"], result)

        # Step 2.6: Store new associations in Firebase
        for association in new_associations:
            doc_ref.add(association.to_firebase())
        
    # Step 3: Create new tasks and reprioritize task list
    new_tasks = task_creation_agent(OBJECTIVE,enriched_result, task["task_name"], [t["task_name"] for t in task_list])

    for new_task in new_tasks:
        task_id_counter += 1
        new_task.update({"task_id": task_id_counter})
        add_task(new_task)
    prioritization_agent(this_task_id)

time.sleep(1)  # Sleep before checking the task list again
