# steamship_integration.py
# Description: This module integrates Steamship API with the task management system

from steamship import Steamship
# from steamship_langchain import LangChainOpenAI
from steamship_package.src.steamship import Steamship
from api import api_key
from config import LLM_model

def create_instance(instance_handle, model_type, custom_parameters):
    instance = Steamship(api_key=api_key)
    instance.model_type = model_type
    instance.instance_handle = instance_handle
    instance.custom_parameters = custom_parameters
    return instance

instances = {}

# Create LLM instances
for model_type, model_instances in LLM_model.items():
    for instance_handle, instance_config in model_instances.items():
        instance = create_instance(instance_handle, model_type, instance_config)
        instances[instance_handle] = instance

# Task queue to hold tasks
task_queue = []

def add_tasks_to_queue(tasks):
    """Add tasks to the task queue."""
    task_queue.extend(tasks)

        
# Access the instances using their handles, e.g.:
gpt4_execution_agent_instance = instances["gpt4-execution-agent-instance"]
gpt4_task_creation_agent_instance = instances["gpt4-task-creation-agent-instance"]
gpt4_task_prioritization_agent_instance = instances["gpt4-task-prioritization-agent-instance"]
gpt4_quality_assurance_agent_instance = instances["gpt4-quality_assurance_agent-instance"]

gpt3_5_software_engineer_instance = instances["gpt3-5-software-engineer-instance"]
gpt3_5_cybersecurity_specialist_instance = instances["gpt3-5-cybersecurity-specialist-instance"]

# Initialize LangChain OpenAI instances for GPT-4 instances
# LC_gpt4_execution_agent = LangChainOpenAI(client=instances["gpt4-execution-agent-instance"])
# LC_gpt4_task_creation_agent = LangChainOpenAI(client=instances["gpt4-task-creation-agent-instance"])
# LC_gpt4_task_prioritization_agent = LangChainOpenAI(client=instances["gpt4-task-prioritization-agent-instance"])

# Initialize LangChain OpenAI instances for GPT-3.5 instances
# LC_gpt3_5_1 = LangChainOpenAI(client=instances["gpt3-5-agent-instance-1"])
# LC_gpt3_5_2 = LangChainOpenAI(client=instances["gpt3-5-agent-instance-2"])
# LC_gpt3_5_3 = LangChainOpenAI(client=instances["gpt3-5-agent-instance-3"])

def gpt4_execution_agent(input_text):
    response = execution_agent_instance.post("execute_task", input_text=input_text)
    result = response["result"]
    return result

def gpt4_task_creation_agent(input_text):
    response = task_creation_agent_instance.post("create_tasks", input_text=input_text)
    new_tasks = response["result"]
    return new_tasks

def gpt4_task_prioritization_agent(input_text):
    response = task_prioritization_agent_instance.post("prioritize_tasks", input_text=input_text)
    prioritized_tasks = response["result"]
    return prioritized_tasks

def gpt4_quality_assurance_agent(input_text):
    response = gpt4_quality_assurance_agent_instance.post("quality_assurance", input_text=input_text)
    result = response["result"]
    return result

def gpt3_5_software_engineer(input_text):
    response = gpt3_5_software_engineer_instance.post("software_engineering", input_text=input_text)
    result = response["result"]
    return result

def gpt3_5_cybersecurity_specialist(input_text):
    response = gpt3_5_cybersecurity_specialist_instance.post("cybersecurity", input_text=input_text)
    result = response["result"]
    return result


