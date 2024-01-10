
# Standard Libraries
import os
import json
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from collections import deque
from typing import Dict, List
import time
import requests

# Suppressing warnings from urllib3
import urllib3
urllib3.disable_warnings()

# .env for environment variables
from dotenv import load_dotenv

# Imports related to LangChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import (
    initialize_agent, Tool, load_tools, AgentType, ZeroShotAgent, Tool, AgentExecutor
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Other utilities and types
from bs4 import BeautifulSoup
from typing import Literal, Dict, Optional, Any, List, Type
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

# Imports related to Splunk
import splunklib.client as client
import splunklib.results as results


# Initial Tasks Creation
'''
Create the first list of tasks
'''
#
# Prompts
#

tasks_initializer_prompt = PromptTemplate(
    input_variables=["objective"],
    template="""
    You are an AI agent responsible for creating a detailed JSON checklist of tasks that will guide other AI agents to complete a given objective.
    Your task is to analyze the provided objective and initial background research and generate a well-structured checklist with a clear starting point and end point,
    as well as tasks broken down to be very specific, clear, and executable by other agents without the context of other tasks. Limit the number of tasks to no more than 4 tasks.

    The current agents work as follows:
    - spl_writer_agent: Writes the intial Splunk SPL snippets.
    - spl_filter_agent: Edits the provided Splunk SPL query. Can also modify existing SPL queries to filter for additional fields to meet the requirements of the task.
    - spl_statistical_analysis_agent: Applies a statistical analysis for the provided Splunk SPL query.
    - spl_refactor_agent: Responsible for refactoring the choosen index, source, and field names for the existing SPL Query to meet the requirements of the task.
    - splunk_executor_agent: Executes Splunk search API queries for tasks.
    - analysis_agent: Responsible for analyzing the results from the of executing the Splunk SPL query

    Here is the detection objective you need to create a checklist for: {objective}.

    To generate the checklist, follow these steps:

    1. Analyze the objective to identify the high-level requirements and goals of the project. This will help you understand the scope and create a comprehensive checklist.

    2. Break down the objective into smaller, highly specific tasks that can be worked on independently by other agents.
    Ensure that the tasks are designed to be executed by the available agents (spl_writer_agent, spl_filter_agent, spl_statistical_analysis_agent, spl_refactor_agent, splunk_executor_agent, and analysis_agent).

    3. Assign a unique ID to each task for easy tracking and organization. This will help the agents to identify and refer to specific tasks in the checklist.

    4. Organize the tasks in a logical order, with a clear starting point and end point.
    The starting point should represent the initial research or understanding necessary for the detection, while the end point should signify the completion of the objective and any finalization steps.

    5. Provide the current context for each task, which should be sufficient for the agents to understand and execute the task without referring to other tasks in the checklist.
    This will help agents avoid task duplication.

    6. Pay close attention to the Windows Event ID, Field Names, and Data and make sure the tasks implement all necessary pieces needed to construct a valid detection.

    7. Compile the tasks into a well-structured JSON format, ensuring that it is easy to read and parse by other AI agents. The JSON should only include fields such as task ID and description.

    REMEMBER EACH AGENT WILL ONLY SEE A SINGLE TASK.
    ASK YOURSELF WHAT INFORMATION YOU NEED TO INCLUDE IN THE CONTEXT OF EACH TASK TO MAKE SURE THE AGENT CAN EXECUTE THE TASK WITHOUT SEEING THE OTHER TASKS OR WHAT WAS ACCOMPLISHED IN OTHER TASKS.

    Make sure tasks are not duplicated.

    Do not take long and complex routes, minimize tasks and steps as much as possible. The final step should always be splunk_executer_agent. Select no more than 7 tasks.

    Here is a sample JSON output for a checklist:

            {{
                "tasks": [
                    {{
                    "id": 1,
                    "description": "Write a Splunk SPL query to detect a <insert> attack",
                    "agent": "spl_writer_agent"
                    }},
                    {{
                    "id": 2,
                    "description": "Edit the existing SPL query to filter for relevant fields such as Windows Event ID, Field Names, and Data",
                    "agent": "spl_filter_agent"
                    }},
                    "id": 3,
                    "description": "Refactor the existing SPL query to ensure the proper index, source, and field names are used",
                    "agent": "spl_refactor_agent"
                    }},
                    "id": 4,
                    "description": "Apply a statistical analysis of the current SPL query using SPL commands such as stats, where, or table to detect patterns indicative of a <insert> attack",
                    "agent": "spl_statistical_analysis_agent"
                    }},

                    {{
                    "id": 5,
                    "description": "Run a Splunk SPL search using the developed SPL query to identify instances of a <insert> attack",
                    "agent": "splunk_executor_agent"
                    }},
                    "id": 6,
                    "description": "Analyze the results of the Splunk search to determine if the attack has occurred",
                    "agent": "analysis_agent"
                    }},
                    ...
                    {{
                    "id": N,
                    "description": "...",
                    }}
                ],
            }}
    The tasks will be executed by either of the three agents:  spl_writer_agent, spl_refactor_agent, splunk_executor_agent, and analysis_agent. ALL tasks MUST start either with the following phrases:
    'Write a Splunk SPL query to...', 'Edit existing SPL to...', 'Run a splunk SPL search to...','Apply a statistical analysis...', or 'Analyze results of...' depending on the agent that will execute the task.
    RETURN JSON ONLY:

    """
)

# Gather Detials for each Task
'''
This will look at the description of the task and add details to the tasks
It will Google if neded but make that decison itself through ReACT chain

Baby Original Prompt:
'''
tasks_details_agent = PromptTemplate(
    input_variables=["objective","task_list_json","detection_procedures","splunk_info", "schema", "detection_procedures"],
    template="""
      You are an AI agent responsible for improving a list of tasks in JSON format and adding ALL the necessary details to each task.
      These tasks will be executed individually by agents that have no idea about other tasks.
      It is FUNDAMENTAL that each task has enough details so that an individual isolated agent can execute.
      The metadata of the task is the only information the agents will have.

      Each task should contain the details necessary to execute it.
      For example, if it creates a function, it needs to contain the details about the arguments to be used in that function and this needs to be consistent across all tasks.

      Look at all tasks at once, and update the task description adding details to it for each task so that it can be executed by an agent without seeing the other tasks and to ensure consistency across all tasks.
      DETAILS ARE CRUCIAL.

      For example, if one task references a Windows EventCode it should have the index and source.
      If another task applies statistical analysis on the event data, it should have the details about the field names and possible values.

      RETURN JSON OUTPUTS ONLY.

      Here is the overall objective you need to refactor the tasks for:
      {objective}.

      Here is the task list you need to improve:
      {task_list_json}

      Here are the current detection procedures from the web you need to reference for the tasks:
      ---(Start detection procedures)---
      {detection_procedures}
      ---(End detection procedures)---

      Here is the index and source information in a list data type format:
      {splunk_info}

      Here are the field names and example values for the given event code. The information is in a dictionary data type format. Ensure that the same fieldname format is chosen as what is shown in the sample data:
      {schema}

      RETURN THE SAME TASK LIST but with the description improved to contain the details you are adding for each task in the list. DO NOT MAKE OTHER MODIFICATIONS TO THE LIST. Your input should go in the 'description' field of each task.

      RETURN JSON ONLY:
    """
)

tasks_details_agent_testing = PromptTemplate(
    input_variables=["objective","task_list_json","detection_procedures", "detection_procedures"],
    template="""
      You are an AI agent responsible for improving a list of tasks in JSON format and adding ALL the necessary details to each task.
      These tasks will be executed individually by agents that have no idea about other tasks.
      It is FUNDAMENTAL that each task has enough details so that an individual isolated agent can execute.
      The metadata of the task is the only information the agents will have.

      Each task should contain the details necessary to execute it.
      For example, if it creates a function, it needs to contain the details about the arguments to be used in that function and this needs to be consistent across all tasks.

      Look at all tasks at once, and update the task description adding details to it for each task so that it can be executed by an agent without seeing the other tasks and to ensure consistency across all tasks.
      DETAILS ARE CRUCIAL.

      For example, if one task references a Windows EventCode it should have the index and source.
      If another task applies statistical analysis on the event data, it should have the details about the field names and possible values.

      RETURN JSON OUTPUTS ONLY.

      Here is the overall objective you need to refactor the tasks for:
      {objective}.

      Here is the task list you need to improve:
      {task_list_json}

      Here are the current detection procedures from the web you need to reference for the tasks:
      ---(Start detection procedures)---
      {detection_procedures}
      ---(End detection procedures)---

      RETURN THE SAME TASK LIST but with the description improved to contain the details you are adding for each task in the list. DO NOT MAKE OTHER MODIFICATIONS TO THE LIST. Your input should go in the 'description' field of each task.

      RETURN JSON ONLY:
    """
)

# Create the Context
'''
This will provide Context about specific fields/events if needed
It will have access to a schema lookup tool for windows event codes.
'''

tasks_context_agent = PromptTemplate(
    input_variables=["objective","task_list_json","detection_procedures"],
    template="""
      You are an AI agent responsible for improving a list of tasks in JSON format and adding ALL the necessary context
      to it from the current detection procedures from the web or any knowledge you have for detection.
      These tasks will be executed individually by agents who have no idea about other tasks or what indexes, sources, or fields exist within the Spunk Database.
      It is FUNDAMENTAL that each task has enough context so that an individual isolated agent can execute. The metadata of the task is the only information the agents will have.

      Look at all tasks at once, and add the necessary context to each task so that it can be executed by an agent without seeing the other tasks.
      Remember, one agent can only see one task and has no idea about what happened in other tasks.
      CONTEXT IS CRUCIAL. For example, if one filters for specific fields within an index and source type.
      The second task should build on that to conduct the calculations or statistics.
      Note that you should identify when calculations or statistics need to happen and specify this in the context.
      Also, you should identify when parts of the SPL command already exist and specify this very clearly because the agents sometimes duplicate things not knowing.

      RETURN JSON OUTPUTS ONLY.

      Here is the overall objective you need to refactor the tasks for: {objective}.
      Here is the task list you need to improve: {task_list_json}
      Here are the current detection procedures from the web you need to reference for the tasks:
      ---(Start detection procedures)---
      {detection_procedures}
      ---(End detection procedures)---


      RETURN THE SAME TASK LIST but with a new field called 'isolated_context' for each task in the list.
      This field should be a string with the context you are adding. DO NOT MAKE OTHER MODIFICATIONS TO THE LIST.

      RETURN JSON ONLY:
    """
  )


#Human Input
'''
Human input context for each task
'''
tasks_human_agent = PromptTemplate(
    input_variables=["task","human_feedback"],
    template="""You are an AI agent responsible for getting human input to improve the quality of tasks in a software project.
    Your goal is to analyze the provided task and adapt it based on the human's suggestions.
    The tasks should  start with either 'Write a Splunk SPL query to...', 'Edit existing SPL to...', 'Run a splunk SPL search to...','Apply a statistical analysis...', or 'Analyze results of...'  depending on the agent that will execute the task.

    For context, this task will be executed by other AI agents with the following characteristics:
    - spl_writer_agent: Writes the intial Splunk SPL snippets.
    - spl_filter_agent: Edits the provided Splunk SPL query. Can also modify existing SPL queries to filter for additional fields to meet the requirements of the task.
    - spl_statistical_analysis_agent: Applies a statistical analysis for the provided Splunk SPL query.
    - spl_refactor_agent: Responsible for refactoring the choosen index, source, and field names for the existing SPL Query to meet the requirements of the task.
    - splunk_executor_agent: Executes Splunk search API queries for tasks.
    - analysis_agent: Responsible for analyzing the results from the of executing the Splunk SPL query

    The current task is:
    {task}

    The human feedback is:
    {human_feedback}

    If the human feedback is empty, return the task as is. If the human feedback is saying to ignore the task, return the following string: <IGNORE_TASK>

    Note that your output will replace the existing task, so make sure that your output is a valid task that starts with one of the required phrases
    ('Write a Splunk SPL query to...', 'Edit existing SPL to...', 'Run a splunk SPL search to...','Apply a statistical analysis...', or 'Analyze results of...' ).

    Please adjust the task based on the human feedback while ensuring it starts with one of the required phrases
    ('Write a Splunk SPL query to...', 'Edit existing SPL to...', 'Run a splunk SPL search to...','Apply a statistical analysis...', or 'Analyze results of...' ).
    Return the improved task as a plain text output and nothing else. Write only the new task."""
  )
# Agent Assignment
'''
Used to assign
'''
task_assigner_agent = PromptTemplate(
    input_variables=["objective","task","recommendation"],
    template=""""You are an AI agent responsible for choosing the best agent to work on a given task.
    Your goal is to analyze the provided major objective of the project and a single task from the JSON checklist generated by the previous agent, and choose the best agent to work on the task.

    The overall objective is: {objective}
    The current task is: {task}

    Use this recommendation to guide you: {recommendation}

    The available agents are:
    - spl_writer_agent: Writes the intial Splunk SPL snippets.
    - spl_filter_agent: Edits the provided Splunk SPL query. Can also modify existing SPL queries to filter for additional fields to meet the requirements of the task.
    - spl_statistical_analysis_agent: Applies a statistical analysis for the provided Splunk SPL query.
    - spl_refactor_agent: Responsible for refactoring the choosen index, source, and field names for the existing SPL Query to meet the requirements of the task.
    - splunk_executor_agent: Executes Splunk search API queries for tasks.
    - analysis_agent: Responsible for analyzing the results from the of executing the Splunk SPL query

    Please consider the task description and the overall objective when choosing the most appropriate agent. Keep in mind that creating a file and writing code are different tasks. If the task involves creating a file, like "calculator.py" but does not mention writing any code inside it, the command_executor_agent should be used for this purpose. The code_writer_agent should only be used when the task requires writing or adding code to a file. The code_refactor_agent should only be used when the task requires modifying existing code.

    In summary, to execute splunk spl, use splunk_executor_agent, to write splunk spl queries, use spl_writer_agent, to modify existing spl, use spl_filter_agent, to apply statistical analysis, use spl_statistical_analysis_agent, to refactor existing spl queries, use spl_refactor_agent, to analyze the results of a splunk search, use the analysis_agent.

    Choose the most appropriate agent to work on the task and return a JSON output with the following format: {{"agent": "agent_name"}}.
    ONLY return JSON output.
    """
  )

# SPL Execution

# SPL Writer
spl_writer_agent = PromptTemplate(
    input_variables=["objective", "task","splunk_info" ,"schema","isolated_context"],
    template="""
      You are a world-class detection engineer and an expert in cyber security,
      threat hunting, and data science.

      For reference, your high level objective is:
      {objective}

      Write the Splunk SPL Query but include explanations/comments.
      Provide no information about who you are and focus on writing the detection query.
      Ensure the query is bug and error free. Respond in a well-formatted markdown with ONLY the SPL code.
      Ensure code blocks are used for the SPL query sections and that the code blocks always start with "spl".

      Approach problems step by step.
      Every query should be formatted to search "All Time" in your Splunk index.
      If there are thresholds in the Splunk SPL query that are used to detect suspicious activity, provide a value you believe would best fit that threshold.
      If there is analysis to be done on a given field use one of the Splunk SPL search commands detect suspicious activity, and provide a value you believe would best fit that threshold.
      Never include comments in the Splunk SPL query or in the code block.

      Your job is to write a Splunk SPL query to accomplish the current task:
      {task}

      For Refrence, sourcetype="WinEventLog" source=WinEventLog:<insert logtype here (System, Security, Application, etc..)>
      Here is the index, source, and sourcetype information in a list data type format:
      {splunk_info}

      Ensure that the current field names in the SPL query match the format from your Splunk server.
      Here are the field names and example values for a given event code. The information is in a dictionary data type format:
      {schema}

      It is important you use this context as a reference for the other pieces of the SPL query that are relevant to your task. PAY ATTENTION TO THIS:
      {isolated_context}
      
      YOUR RESPONSE:
    """
)
spl_writer_agent_testing = PromptTemplate(
    input_variables=["objective", "task","isolated_context"],
    template="""
      You are a world-class detection engineer, an expert in cyber security,
      threat hunting, and data science.

      For reference, your high level objective is to:
      {objective}

      Write the Splunk SPL Query but include explanations/comments.
      Provide no information about who you are and focus on writing the detection query.
      Ensure the query is bug and error free. Respond in a well-formatted markdown with ONLY the SPL code.
      Ensure code blocks are used for the SPL query sections and that the code blocks always start with "spl".

      Approach problems step by step.
      If there are time based thresholds in the Splunk SPL query that are used to detect suspicious activity, provide a value you believe would best fit that threshold.
      If there is analysis to be done on a given field use one of the Splunk SPL search commands detect suspicious activity, and provide a value you believe would best fit that threshold.
      Never include comments in the Splunk SPL query or in the code block.

      Your job is to write a Splunk SPL query to accomplish the current task:
      {task}

      Reference the following information from google when building your query:
      {isolated_context}

      Respond with only a plain text string containing the SPL needed to complete the task nothing else. IMPORTANT: JUST RETURN SPL QUERY, YOUR OUTPUT WILL BE ADDED DIRECTLY TO THE SEARCH BY OTHER AGENT. BE MINDFUL OF THIS
      --- (Example SPL Response START) --- 
      index=main sourcetype=WinEventLog EventCode=7045
      --- (Example SPL Response END) ---
      
      YOUR RESPONSE:
    """
)
spl_filter_agent = PromptTemplate(
    input_variables=["objective", "task","previous_query", "isolated_context"],
    template="""
      You are a world-class detection engineer, an expert in cyber security,
      threat hunting, and data science.

      For reference, your high level objective is to:
      {objective}

      Update the provided Splunk SPL Query but include explanations/comments.
      Provide no information about who you are and focus on writing the detection query.
      Ensure the query is bug and error free. Respond in a well-formatted markdown with ONLY the SPL code.
      Ensure code blocks are used for the SPL query sections and that the code blocks always start with "spl".

      Approach problems step by step.
      If there are time based thresholds in the Splunk SPL query that are used to detect suspicious activity, provide a value you believe would best fit that threshold.
      If there is analysis to be done on a given field use one of the Splunk SPL search commands detect suspicious activity, and provide a value you believe would best fit that threshold.
      Never include comments in the Splunk SPL query or in the code block.

      Your job is to update the provided Splunk SPL query to accomplish the current task:
      {task}

      Here is the provided Splunk SPL query:
      {previous_query}

      Reference the following information from google when building your query:
      {isolated_context}

      Respond with only a plain text string containing the SPL needed to complete the task nothing else. IMPORTANT: JUST RETURN SPL QUERY, YOUR OUTPUT WILL BE ADDED DIRECTLY TO THE SEARCH BY OTHER AGENT. BE MINDFUL OF THIS
      --- (Example SPL Response START) --- 
      index=main sourcetype=WinEventLog EventCode=7045
      --- (Example SPL Response END) ---

      YOUR RESPONSE:
    """
)
spl_statistical_analysis_agent = PromptTemplate(
    input_variables=["objective", "task","previous_query", "isolated_context"],
    template="""
      You are a world-class detection engineer, an expert in cyber security,
      threat hunting, and data science.

      For reference, your high level objective is to:
      {objective}

      Update the provided Splunk SPL Query but include explanations/comments.
      Provide no information about who you are and focus on writing the detection query.
      Ensure the query is bug and error free. Respond in a well-formatted markdown with ONLY the SPL code.
      Ensure code blocks are used for the SPL query sections and that the code blocks always start with "spl".

      Approach problems step by step.
      If there are time based thresholds in the Splunk SPL query that are used to detect suspicious activity, provide a value you believe would best fit that threshold.
      If there is analysis to be done on a given field use one of the Splunk SPL search commands detect suspicious activity, and provide a value you believe would best fit that threshold.
      Never include comments in the Splunk SPL query or in the code block.

      Your job is to update the provided Splunk SPL query to accomplish the current task:
      {task}

      Here is the provided Splunk SPL query:
      {previous_query}

      Reference the following information from google when building your query:
      {isolated_context}

      Respond with only a plain text string containing the SPL needed to complete the task nothing else. IMPORTANT: JUST RETURN SPL QUERY, YOUR OUTPUT WILL BE ADDED DIRECTLY TO THE SEARCH BY OTHER AGENT. BE MINDFUL OF THIS
      --- (Example SPL Response START) --- 
      index=main sourcetype=WinEventLog EventCode=7045
      --- (Example SPL Response END) ---

      YOUR RESPONSE:
    """
)

# SPL Refactor
'''
Outside the for loop last step is the summary
and it will be the final output
takes results from all the SPL Coding stuff/all tasks in loop
'''
spl_refactor_agent = PromptTemplate(
    input_variables=["objective","task_description", "isolated_context","existing_spl","command_execution_errors"],
    template="""
      You are an AI agent responsible for refactoring Splunk SPL queries to accomplish a given task.
      Your goal is to analyze the provided major objective of the project, and the task description and refactor the code accordingly.
      If there are any errors listed below that will be your number 1 priority to fix. However, your main goal is to review the  improve

      For reference, your high level objective is {objective}
      The current task description is: {task_description}

      Existing SPL you should refactor:
      {existing_spl}

      To help you make the SPL useful for this detection, use this context as a reference to determine if there are any fields to analyze or statistical calculations added to the SPL query:
      {isolated_context}

      Errors from previous SPL:
      {command_execution_errors}

      Based on the task description, objective, and isolated context, refactor the existing SPL query to achieve the task. Make sure the refactored query is relevant to the task and objective, follows best practices, etc.

      Respond with only a plain text string containing the SPL needed to complete the task nothing else. IMPORTANT: JUST RETURN SPL QUERY, YOUR OUTPUT WILL BE ADDED DIRECTLY TO THE SEARCH BY OTHER AGENT. BE MINDFUL OF THIS
      --- (Example SPL Response START) --- 
      index=main sourcetype=WinEventLog EventCode=7045
      --- (Example SPL Response END) ---

      YOUR RESPONSE:
    """
)

#fix_bad_fields
spl_normalize_agent = PromptTemplate(
    input_variables=["existing_spl","objective", "splunk_info","schema"],
    template="""
    You are a world-class detection engineer and an expert in cyber security, threat hunting, and data science.
    You have been provided an SPL query that does not return any results due to one or more of the field names used not matching the field names in your Splunk server.
    Ensure that the SPL query starts with index=
    If the current SPL query starts with anything other than index, remove those strings from your final splunk query.
    Correct the field names in the SPL query to match the fields in your Splunk server.

    The current SPL query provided:
    {existing_spl}

    Ensure the index and source selected contain the data to answer the objective:
    {objective}

    Ensure the selected fields match the information in your current Splunk server or else the query will not return any data.
    You ran two Splunk queries to retrieve the index, source, and fields for a given event code in your Splunk server and the information was returned.

    Select the index and source that will contain the information from your Splunk SPL query.
    For reference here is the information from your Splunk server:
    {splunk_info}

    Ensure the current fields match the fields from your splunk server for the given event code. For reference here are the fields in a list format from your Splunk server for each event code:
    {schema}

    Only modify the index, source, sorucetype and field names to match the information from your Splunk server. Do not modify the value inside the fields.
    
    Respond with only a plain text string containing the SPL needed to complete the task nothing else. IMPORTANT: JUST RETURN SPL QUERY, YOUR OUTPUT WILL BE ADDED DIRECTLY TO THE SEARCH BY OTHER AGENT. BE MINDFUL OF THIS
    --- (Example SPL Response START) --- 
    index=main sourcetype=WinEventLog EventCode=7045
    --- (Example SPL Response END) ---

    YOUR RESPONSE:
    """
)


# Research Agent
'''
Add come background research
'''
research_system_template = SystemMessage(
    content="""
    You are a world class researcher, who can do detailed research on windows attacks and produce facts based detection procedures using only windows security event logs;
    you do not make things up, you will try as hard as possible to gather facts & data to back up the research;
    you will priortize searching for specific event log id, field names, and values for field names

    Write a comprehensive guide to build a Splunk SPL Query including event codes and field names.
    DO NOT build any actual SPL Query.

    Please make sure you complete the objective above with the following rules:
    1/ You should do enough research to gather as much information as possible about the objective
    2/ If there are url of relevant links & articles, you will scrape it to gather more information
    3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
    4/ You should not make things up, you should only write facts & data that you have gathered
    5/ Do not make things up, you should only write facts & data that you have gathered
    """
)


event_id_prompt = PromptTemplate(
    input_variables=["detect_procedure"],
    template="""
    Your task is to assist users in building comprehensive Splunk SPL Queries tailored to their specific needs. You will focus on extracting the top two (2) relevant Event ID to be incorporated into the SPL query, ensuring that the detection logic is accurately represented. DO NOT build any actual SPL Query. You can analyze detection procedure and capable of EventCode Extraction.

    EventCode Extraction: Understand the user's detection logic and extract or recommend the top two (2) relevant Windows Event ID that need to be included in the SPL query. Only Extract the two most relevent Event IDs.

    Here is the current detection procedure:
    --- (Start Detection procedure) ---
    {detect_procedure}
    --- (End Detection procedure) ---
    Your output must be a list of EventCode numbers. Here is an example:
    [4769, 4688]
    RETURN LSIT ONLY:
    """
)
# Analyze Results
summarize_splunk_results = PromptTemplate(
    template="""You are a world-class detection engineer and an expert in cyber security,
    threat hunting, and data science. You have been asked to perform the following objective:
    {objective}.

    The query you created was the following:
    {query}

    The results of the query are:
    {results}

    Provide a summary to your manager to describe the results of the query based on the goal provided. You do not need to provide a summary of the query, only provide a brief summary of the result and how you would recommend the security team should respond.
    Be as clear and concise as possible when describing the results of the query and the recommendation you provide. If possible, ensure your summary includes number of affected hosta and host names.
    """,
    input_variables=["objective", "query", "results"],
)
# Listen to the smart human
splunk_human_input_agent = PromptTemplate(
    template="""You are a world-class detection engineer and an expert in cyber security,
    threat hunting, and data science.

    You have created an amazing Splunk SPL query but have been asked to modify the command by your superior:
    {human_input}.

    The query you created was the following:
    {query}

    If your superior says "No updates required" or a variation of that comment to signify the Splunk SPL query does not need modification, then simply return the Splunk SPL query you created.

    If your superior provides instructions for modifying the Splunk SPL query, modify your Splunk SPL query and respond with only the SPL query needed to complete the task. Do not explain the query. Do not add comments.
    YOUR RESPONSE:
    """,
    input_variables=["human_input", "query"],
)