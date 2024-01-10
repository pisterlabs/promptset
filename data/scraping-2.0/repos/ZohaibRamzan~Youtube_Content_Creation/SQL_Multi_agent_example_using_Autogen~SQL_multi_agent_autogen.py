import os
from dotenv import load_dotenv
import openai
import autogen
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import query_maker_gpt_system_prompt, admin_prompt, data_engineer_prompt
from langchain.prompts import PromptTemplate
import mysql.connector

#load env variables
load_dotenv()

# openai api key
api_key=os.getenv('OPENAI_API_KEY')

# Set your LLms Endpoint
config_list_gpt_turbo = autogen.config_list_from_models(model_list=[ "gpt-4"])

def query_maker(user_input):
    # make sql queries using LLM chain
    openaiLLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7,
                                        openai_api_key=api_key, cache=False)
    prompt_template = PromptTemplate.from_template(
        "{system_prompt} + '\n' +  {user_input}.")

    chain = LLMChain(llm=openaiLLM, prompt=prompt_template)
    query=chain.run({"system_prompt": query_maker_gpt_system_prompt, "user_input": user_input})
    return query

def run_sql_query(sql_query):
    config = {
        'host': os.getenv('azure_sql_host'),
        'user': os.getenv('user'),
        'password': os.getenv('password'),
        'database': os.getenv('database'),
        'ssl_ca': os.getenv('ssl_ca')  # Path to the SSL CA certificate (optional)
    }

    conn = mysql.connector.connect(**config)

    cursor = conn.cursor()
    try:
        # Execute SQL queries
        cursor.execute(sql_query)
        result = cursor.fetchall()
    except Exception as e:
        return e

    return result

gpt_turbo_simple_config = {
    "use_cache": False,
    "temperature": 0.5,
    "config_list": config_list_gpt_turbo,
    "request_timeout": 90}


gpt_turbo_config = {
    "use_cache": False,
    "temperature": 0.7,
    "config_list": config_list_gpt_turbo,
    "request_timeout": 90,
    "functions" : [
    {
        "name": "query_maker",
        "description": "generates sql query as per user input",
        "parameters": {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "This is the input from the user side.",
                }
                ,
            },
            "required": ["user_input"],
        },
    },

{
        "name": "run_sql_query",
        "description": "This function is used to run sql query against user input to get the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "This is the mysql query.",
                }
                ,
            },
            "required": ["sql_query"],
        },
    }



    ]
}
function_map={"query_maker": query_maker ,"run_sql_query": run_sql_query}
termination_msg="If everything looks good, respond with Approved."

def is_termination_msg(content):
    have_content=content.get("content", None) is not None
    if have_content and "Approved" in content["content"]:
        return True
    else:
        return False


user_proxy = autogen.UserProxyAgent(
   name="Admin",
   system_message= admin_prompt + termination_msg,
   human_input_mode="Never",
    is_termination_msg=is_termination_msg


)

engineer = autogen.AssistantAgent(
    name="Data_Engineer",
    llm_config=gpt_turbo_config,
    system_message=data_engineer_prompt + termination_msg,
    function_map=function_map
)

# register the functions
user_proxy.register_function(function_map={"query_maker": query_maker ,"run_sql_query": run_sql_query},)

user_proxy.initiate_chat( engineer,
    message="""How many Apple pro vision we have in our stock?""", clear_history=True
)

user_proxy_chat=user_proxy.chat_messages
engineer_chat=engineer.chat_messages

