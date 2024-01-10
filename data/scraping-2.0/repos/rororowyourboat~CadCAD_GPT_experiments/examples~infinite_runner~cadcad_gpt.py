import openai
import json
import os

import pandas as pd
from radcad import Experiment
from radcad.engine import Engine
#importing radcad model from models folder
from infinite_runner_radcad import model, simulation, experiment

from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
# from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

with open('docs.txt', 'r') as file:
    docs = file.read().replace('\n', '')

##########################
# Tool kit
# tools in the tool kit
df = pd.DataFrame(experiment.run())

def change_param(param,value):
    '''Changes the value of a parameter in the model'''
    # simulation.model.initial_state.update({
    # })
    value = float(value)
    simulation.model.params.update({
        param: [value]
    })
    experiment = Experiment(simulation)
    experiment.engine = Engine()
    result = experiment.run()
    # Convert the results to a pandas DataFrame
    globals()['df'] = pd.DataFrame(result)
    return f'new {param} value is {value} and the simulation dataframe is updated'

def model_info(param):
    '''Returns the information about the model'''
    if param == 'all':
        return simulation.model.params
    elif param in simulation.model.params:
        return f'{param} = {simulation.model.params[param]}'
    else:
        return f'{param} is not a parameter of the model'

# pandas agent as a tool

def analyze_dataframe(question):
    '''Analyzes the dataframe and returns the answer to the question'''
    # pandas_agent = agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    answer = pandas_agent.run(question)
    
    return answer

def model_documentation(question):
    '''Returns the documentation of the model'''
    vectorstore = FAISS.from_texts([docs], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | model 
        | StrOutputParser()
    )
    info = chain.invoke(question)

    return info


def A_B_test(param,param2,metric):
    '''Runs an A/B test on the model'''

    return 'A/B test is running'


# tool descriptions

function_descriptions_multiple = [
    {
        "name": "change_param",
        "description": "Changes the parameter of the cadcad simulation and returns dataframe as a global object. The parameter must be in this list:" + str(model.params.keys()),
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "parameter to change. choose from the list" + str(model.params.keys()),
                },
                "value": {
                    "type": "string",
                    "description": "value to change the parameter to, eg. 0.1",
                },
            },
            "required": ["param", "value"],
        },
    },
    {
        "name": "model_info",
        "description": "quantitative values of current state of the simulation parameters. If no param is specified the argument should be 'all'",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "type of information to print. choose from the list: " + str(model.params.keys()),
                },
            },
            "required": ["param"],
        },
    },
    {
        "name": "analyze_dataframe",
        "description": "Use this whenever a quantitative question is asked about the dataframe. The question should be taken exactly as asked by the user",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asked by user that can be answered by an LLM dataframe agent",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "model_documentation",
        "description": "use when asked about documentation of the model has information about what the model is, assumptions made, mathematical specs, differential model specs etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asked by user that can be answered by an LLM dataframe agent",
                },
            },
            "required": ["question"],
        },
    }
]

##################

# Agents

def planner_agent(prompt):
    """Give LLM a given prompt and get an answer."""

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
            "role": "system",
            "content": '''
            You will be provided with a question by the user that is trying to run a cadcad python model. Your job is to provide the set of actions to take to get to the answer using only the functions available.
            These are the functions available to you: {function_descriptions_multiple}. always remember to start and end plan with ###. Dont give the user any information other than the plan and only use the functions to get to the solution.

            User: whats the current value of xyz?
            Planner: ### 1) we use the function model_info to fetch the xyz parameter ###
            User: What is the current value of all params?
            Planner: ### 1) we use the function model_info to fetch all the parameters ###
            User: What are the assumptions in this model?
            Planner: ### 1) use the function model_documentation to fetch the assumptions in this model. ###
            User: What are the metrics and params in the model?
            Planner: ### 1) use the function model_documentation to fetch the metrics and params in the model. ###
            User: What are the columns in the dataframe?
            Planner: ### 1) use the function analyze_dataframe to fetch the columns in the dataframe. ###
            User: What would happen to the A column at the end of the simulation if my xyz param was 20?
            Planner: ### 1) we use function change_param to change the xyz parameter to 20 .\n 2) we use function analyze_dataframe to get the A at the end of the simulation. ###
            USer: What is the current value of my xyz param? can you change it to 50 and tell me what the A column at the end of the simulation would be?
            Planner: ### 1) we use function model_info to fetch the crash_chance parameter. \n 2) we use function change_param to change the xyz parameter to 50 .\n 3) we use function analyze_dataframe to get the A at the end of the simulation. ###
            '''
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
    )

    output = completion.choices[0].message
    return output

def executor_agent(prompt):
    """Give LLM a given prompt and get an answer."""

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": prompt}],
        # add function calling
        functions=function_descriptions_multiple,
        function_call="auto",  # specify the function call
    )

    output = completion.choices[0].message
    return output

######################
# utils 

# plan parser function which takes a string and returns a list of functions to call. It uses the \n as a delimiter to split the string into a list of functions to call.
def plan_parser(plan):
    plan = plan.split('###')[1]
    plans = plan.split('\n')
    # plans = [x.strip() for x in plans]
    #strip the blank space before and after the sentences
    # plans = [x.strip() for x in plans if x.strip() != '']  
    return plans


# pritn with colors
def print_color(string, color):
    print("\033["+color+"m"+string+"\033[0m")


#######################
# orchestration pipeline

# def orchestrator_pipeline(user_input):
#     plan = planner_agent(user_input).content
#     plan_list = plan_parser(plan)
#     print_color("Planner Agent:", "32")
#     print('I have made a plan to follow: \n')

#     for plan in plan_list:
#         print(plan)

#     print('\n')
#     for plan in plan_list:
#         print_color("Executor Agent:", "31")
#         print('Thought: My task is to', plan)
#         answer = executor_agent(plan)
#         print('Action: I should call', answer.function_call.name,'function with these' , json.loads(answer.function_call.arguments),'arguments')
#         if answer.function_call.name == 'analyze_dataframe':
#             print_color("Analyzer Agent:", "34")
#         print('Observation: ', eval(answer.function_call.name)(**json.loads(answer.function_call.arguments)))


def cadcad_gpt(user_input):
    plan = planner_agent(user_input).content
    plan_list = plan_parser(plan)
    print_color("Planner Agent:", "32")
    print('I have made a plan to follow: \n')

    for plan in plan_list:
        print(plan)

    print('\n')
    for plan in plan_list:
        
        print_color("Executor Agent:", "31")
        print('Thought: My task is to', plan)
        answer = executor_agent(plan)
        print('Action: I should call', answer.function_call.name,'function with these' , json.loads(answer.function_call.arguments),'arguments')
        if answer.function_call.name == 'analyze_dataframe':
            print_color("Analyzer Agent:", "34")
        print('Observation: ', eval(answer.function_call.name)(**json.loads(answer.function_call.arguments)))
    

# user_prompt = "whats the current value of crash chance?"
# print(executor_agent(user_prompt))