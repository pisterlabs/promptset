from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from utils.parsers import *


PAST_MISTAKES ='''

Below I have mentioned common mistakes made by you while using the tools.

{mistakes}

!! PLEASE GO THROUGH THEM CAREFULLY AND AVOID MAKING SIMILAR MISTAKES.
Note that the context of above mistakes are different AND INDEPENDENT FROM CURRENT USER QUERY.
DO  NOT TAKE CONTEXT FROM ABOVE QUERIES.

'''
PREFIX = """
Below are the tools in your tool-kit along with their description to help you decide on tool choice.
"""

#____________________________________________________________________________________________________________
FORMAT_INSTRUCTIONS = """
ALERT !!!
  - The Thought-Action-Observation repeats until we feel that agent has completely answered the user query.
  - Each time this process repeates, you need to write some reason in Thought of choosing a particular tool.

Use the following format:

Question: the input question you must answer

Thought : The reason of picking the tool in process of answering user query.

Action : the Tool to take , should be one of [{tool_names}]

Action Input: - Your selected tool will need get its arguments filled by another agent. This agent does not have access to the query or the current output chain. 
              - PRECISELY TELL THIS AGENT HOW YOU WANT IT TO FILL THE ARGUMENTS, in natural language, give emphasis to the argument name and its value.
              - IF you feel that this tool should needs output of other tools, you can infer their output stored in format $$PREV[i], where i is the index of the output you want to use.

... (this Thought/Action/Action Input must ONLY OCCUR ONCE)


Note that it is possible that the query has been successfully answered and no further tool calls are required
In this case return:
Thought: Task has been completed
Action: NONE
Action Input: Task complete
"""

# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================

# MISTAKE_SELECTION =  '''

# Below you are provided with one of the past mistakes made by another AI agent on some other user query :
# {mistake}

# Below you are provided with the current user query :
# CURRENT_USER_QUERY : {input}

# Check diligently if the current user query is similar to the past query, in terms of the vulnerability of the AI agent to make the same mistake.
# If there are some chances of making the same or similar mistake on the current user query, return 1 else return 0.

# ANSWER : 
# '''

MISTAKE_SELECTION =  '''
As a person learns to perform a task correctly by acknowledging his mistakes, similarly an AI agent can also learn to perform a task correctly by acknowledging its mistakes.

Below is the user query AI agent is trying to solve:
USER_QUERY : {input}

Below is one of the past mistake made by AI agent on some other user query:
MISTAKE : {mistake}

- You need to check if the above mistake is relevant to the current user query or not. 
- Whether the AI agent should use this mistake as experience while solving the current user query or not.

FORMAT_INSTRUCTIONS :
- Return 1 if the above mistake is relevant to the current user query, else return 0.
- Stick to the above information provided, decide importance of mistake judiciously, don't pollute the information.

ANSWER :
'''

# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================

TOOL_INPUT_PROMPT = '''
The user query is as follows:
  User_Query : {query}

The intermediate steps taken till now to solve the above query are as follows:
{intermediate_steps}

Below is the next tool that needs to be used as next tool in intermediate steps: {tool_name}

The short description of the tool, to help you reason out, is as follows:
{tool_description}

You are expected to create a sub-task for the above tool from the given user_query, tool_description and the intermediate steps taken till now.

While creating the sub-task for above tool, adhere to tool description. 
Don't query tool for tasks which are not mentioned in tool description.

FORMAT INSTRUCTIONS :
{format_instructions}
'''
# ===================================================================================================================================================================================================


EXAMPLES = [
    {
      'query': "Find all high priority issues related to part 'FEAT-123' created by user 'DEVU-123', prioritize them, and add them to the current sprint" , 
      'intermediate_steps': '''[
                              {{"tool_name": "works_list", "arguments": [{{"argument_name": "issue.priority", "argument_value": "high"}}, 
                                            {{"argument_name": "applies_to_part", "argument_value": "FEAT-123"}}, 
                                            {{"argument_name": "created_by", "argument_value": "DEVU-123"}}]}},
                              {{"tool_name": "prioritize_objects", "arguments": [{{"argument_name": "objects", "argument_value": "$$PREV[0]"}}]}},
                              {{"tool_name": "get_sprint_id", "arguments": []}},
                            ]''',

    'tool_name': 'add_work_items_to_sprint',
    'tool_description': "Adds the given work items to the sprint. This tool needs to know the list of work_id and the sprint_id to which the work items should be added.",
    'tool_input': "Add work items $$PREV[1] to sprint_id $$PREV[2]"
  
  }
]


EXAMPLE_FORMATTER_TEMPLATE = """
query: {query}\n
intermediate steps : {intermediate_steps}\n
tool_name: {tool_name}
tool_description: {tool_description}

tool_input: {tool_input}\n\n
"""

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["query", "intermediate_steps" , "tool_name" , "tool_description", "tool_input"],
    template=EXAMPLE_FORMATTER_TEMPLATE,
)

# ===================================================================================================================================================================================================

sub_task_prompt = FewShotPromptTemplate( 
    examples=EXAMPLES,

    # prompt template used to format each individual example
    example_prompt=EXAMPLE_PROMPT,

    # prompt template string to put before the examples, assigning roles and rules.
    prefix="Here are some few shots of how to create sub-task for a given tool based query, intermediate_steps and tool_description:\n",
    
    # prompt template string to put after the examples.
    suffix=TOOL_INPUT_PROMPT,
    
    # input variable to use in the suffix template
    input_variables=["query" , "intermediate_steps" , "tool_name" , "tool_description"],
    example_separator="\n", 
    partial_variables= {'format_instructions' : sub_task_parser.get_format_instructions()}
)


# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================

MISSED_TOOL_TEMPLATE = '''
There is an AI agent which is picking up tools under ReAct framework to solve user queries.
It misses picking up correct tool, being unable to reason out its usage for the given query.

I provide you some few shots how to reason out the mistake highlight of a tool based on the user query and tool description:

Example_Query : "Prioritize my p0 issues"
Tools missed: 
Incorrect tool uses:
Example_MISTAKE : "The tool 'who_am_i' is useful as there is a keyword 'my' which hints towards the user currently logged in. So, this tool can get the user id of the user currently logged in." 

Example_Query : "Summarize high severity tickets from the customer UltimateCustomer"
Tools missed:
Incorrect tool uses:
Example_MISTAKE :"We need to find the id of the object, so we must use the tool 'search_object_by_name' tool which searches for the object id based on the name of the object."


Example_Query : "What are my all issues in the triage stage under part FEAT-123? Summarize them"
Tools missed:
Incorrect tool uses:
Example_MISTAKE :"We need to find the id of the object, so we must use the tool 'search_object_by_name' tool which searches for the object id based on the name of the object."

- You need to provide an eye-catchy insight of why that tool should not be missed for the given query based on the user query and tool description. 
- You insight will help the agent to learn from its mistakes. Don't be super-specific to user query, keep the tool description in consideration. 
- Keep your insight within 20 words and at least 9 words. Present answer in a paragraph.

USER_QUERY : {query}
Tools missed : {tools_missed}
Incorrect tool uses : {incorrect_tool_uses}
Agent_MISTAKE : 
'''

missed_tool_prompt = PromptTemplate(template=MISSED_TOOL_TEMPLATE, input_variables=['agent_scratchpad','query' ,'correct_tool_name' , 'tool_description'])

# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================

TOOLS_PROMPT_EXAMPLES = '''

Your task is to extract the argument value from user query based on argument description and argument type.

The datatype is {arg_dtype}

The argument description is:
{arg_description}

The above mentioned arguments have their values present in the query. You need to extract the argument value from the query.
Don't pollute the information, stick to information provided in the user query.

Your query is:
{user_query}

FORMAT INSTRUCTIONS --->
  - Don't return anything else other than the argument value.
  - Ensure that the argument value is in correct data type before returning.
  - If the argument value is not explicitly present in the query, then return "NONE".

ALERT !!!
- If the Query contains specific keywords like $$PREV[i], where i is the index of the output you want to use, 
          then it is a symbolic representation of the output and is NOT THE ACTUAL OUTPUT
- use $$PREV[i] as whole and don't pass invalid representation like "$$PREV" OR "$$PREV[]" or i

ANSWER :
'''

#____________________________________________________________________________________________________________

ARG_FILTER_PROMPT = '''
I want to use a tool which has a lots of arguments. I want to filter out only those arguments which can surely be extracted from the user query.

Below I provide the arguments that the tool can take along with their description:
{arg_description}

{format_instructions}

Below I provide the query, which needs to be referenced whiele deciding which arguments to filter out:
QUERY : {query}

ALERT !!!
- Don't create any new argument from user query, make sure that filtered argument have correct name.
- If the Query contains specific keywords like $$PREV[i], then take it as a whole.
- Stick to information provided in the user query and description of arguments.
- Don't pollute the argument filtering with any assumptions.
- Make sure that there are no parsing errors.
'''

#____________________________________________________________________________________________________________

LOGICAL_TEMPLATE = '''
You need to return a code block executing the above user query in the python.

Below I provide an example code block in python so that you know the desired output format:
Example 1:
```
### Query: Calculate the difference in count between P1 and P2 issues

def count_difference(p1_issues, p2_issues):
    return len(p1_issues) - len(p2_issues)
count_difference("$$PREV[0]", "$$PREV[1]")
```

Example 2:
```
### Query: Extract the first five tasks from the list of all tasks.

def first_five_tasks(tasks):
    return tasks[:5]

first_five_tasks("$$PREV[0]")
```
(note that there were other tool calls before this as well that were ommitted)

You may devise any function of your own using a combination of sum, variables, loops, if-else statements etc.

- Make sure that the code is in correct syntax before returning.
- Don't return anything else other than the code block. 
- Simply return the code block, nothing else to be returned

You have the following user query:
{query}
'''
# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================
CRITIQUE_TEMPLATE = '''

Below you are provided the tools available in toolkit and their description :
{tools}

FORMAT INSTRUCTIONS :
{format_instructions}

Here are a few examples of sample outputs:

QUERY_EXAMPLE : What is the use of life?
OUTPUT : {{"answer" : 0 , "reason" : "The available tools are not useful to answer the query."}}

QUERY_EXAMPLE : "List all work items similar to TKT-420 and add the top 2 highest priority items to the current sprint
OUTPUT : {{"answer" : 1 , "reason" : "We can use get_similar_items, prioritize_objects, get_sprint_id, add_work_items_to_sprint to solve query."}}


QUERY_EXAMPLE : Search for youtube videos of user id DEVU-67
OUTPUT : {{"answer" : 0 ,'reason' : "no tool is present to search for youtube videos"}}


QUERY_EXAMPLE : "Create a excel file of work items in the current sprint
OUTPUT : {{"answer" : 0 , "reason" : "no tool is present to create excel file"}}

Give a similar answer, reason pair for the below query. If answer is 1, tell me what all tools you would use

QUERY : {query}

ALERT !!
- Make sure that there are no parsing errors.

'''

critique_prompt = PromptTemplate(template=CRITIQUE_TEMPLATE, input_variables=['query' ,'tools'], 
                                                      partial_variables={'format_instructions' : critique_parser.get_format_instructions()})

# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================
# ===================================================================================================================================================================================================

# You are also provided the dataypes of arguments present in the user query:
# {function_signature}


# ["red" , ""$$PREV[0]"]