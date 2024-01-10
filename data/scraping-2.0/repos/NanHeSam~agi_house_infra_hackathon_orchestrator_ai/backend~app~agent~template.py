from langchain.prompts import ChatPromptTemplate

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
prompt_template = ChatPromptTemplate.from_template(template_string)
# customer_messages = prompt_template.format_messages(
#                     style=customer_style,
#                     text=customer_email)

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Orchestrate planning + execution.
# The goal is to have an agent at the top-level (e.g. so it can recover from errors and re-plan) while
# keeping planning (and specifically the planning prompt) simple.
TASK_BREAKDOWN_PROMPT = """You are an agent that assists with user to generate a plan based on existing tools, but you don't have to use all of them.
You should breakdown the user query into one or multiple tasks abased on the avaliable tools and their descriptions. 

Here are the tools that are available and their descriptions: {tool_descriptions}


Starting below, you should follow this format:

User query: the query a User wants help
Thought: you should always think about what to do
Tool: the tool to use. The tool should be one of the tools [{tool_names}]
Goal: what task can this tool help with and why you choose this tool.
... (this Tool/Goal can repeat N times)
Thought: I am finished planning a task and have the information the user asked for or the data the user asked to create


Example:
User query: can you help me plan a 3 day trip in new york?
Thought: I should break it down into smaller tasks and find the tools for them.
Tool: google_search
Goal: search for sightseeing in the desitination
Tool: hotel_search
Goal: search for avaialble hotels and pricing
Thought: I have breakdown into 3 subtasks find sighseeing, find hotel, find flight and found tools for each tasks.
...

Begin!

User query: {input}
Thought: I should generate a plan to help with this query.
"""

SIMPLE_TASK_BREAKDOWN_PROMPT = """
You are an agent that assists with user to break down a request into atomic tasks based on the context of user preferences. if user preference is emtpy, you can ignore that.

Starting below, you should follow this format:

User query: the query a User wants help
Thought: you should always think about what to do
Step1: Do task 1 
Step2: Complete task 2
Step3: Finish task 3
...
Thought: I am finished planning tasks and have the information the user asked for or the data the user asked to create

Begin!
User query: {input}
User preference: {preference}
Thought: I should generate a plan to help with this query.

"""


FIND_REASON_PROMPT = """
You are an assistant that help user to reason how he should use certain tool to achieve a certain task. the tool description and the task will be given.

Tool description: {tool_description}

Task: {task_description}
"""

EXPLAIN_MATCHING_PROMPT = """
How can I use the tool to achieve {task}?
"""

TOOL_PATTERN = r'Tool:\s*(.*?)\n'
GOAL_PATTERN = r'Goal:\s*(.*?)\n'
