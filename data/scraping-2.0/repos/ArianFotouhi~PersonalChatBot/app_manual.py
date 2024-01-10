import json
import openai
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import sqlite3
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import my_api_key, model_name


openai.api_key = my_api_key
memory_length = 20
history = []

debugger_print = False

def get_flight_info(origin, destination):

    flight_info = {
        "origin": origin,
        "destination": destination,
        "datetime": str(datetime.now()),
        "airline": "Qatar Airways",
        "flight": "Q491",
    }

    return json.dumps(flight_info), None


def db_query(user_prompt):
    if debugger_print:
        print('I AM IN DB_QUERY')
    
    cust_history = chat_history_customizer(['bot shown reply', 'bot chosen function (hidden to user)'])
    
    if debugger_print:
        print('history in dbquery', cust_history[:2])
    
    prompt = (
        PromptTemplate.from_template(
            """
            your Human user question is: {user_prompt} with conversation history of:
            {cust_history}
            ONLY return the {sql_type} query without extra words show it like eg SELECT * FROM ...
            """)
        + 
        "\n\n , Query assumption:  I have my tables (key) and columns (their value) as {table_info}"
    )

    model = ChatOpenAI(temperature=0, openai_api_key= my_api_key)
    chain = LLMChain(llm=model, prompt= prompt)

    ans = chain.run(user_prompt = user_prompt, sql_type= "SQLite", table_info = table_info, cust_history = cust_history[:2])
#    print('sql in function', ans)
    try:

        con = sqlite3.connect("chinook.db")
        cur = con.cursor()
        cur.execute(ans)

        tables = cur.fetchall()

#        for table in tables:
#            print(table)

        #Close the cursor and the connection
        cur.close()
        con.close()
        return f'Query result for {user_prompt} is: '+str(tables), ans

    except Exception as e:
        if debugger_print:
            print('Sorry the search was unsuccessful, could you please try again with more specific information')
            print(e)
        return None, 'Query was not created'


def get_tables_and_columns_sqlite(connection):
        tables_columns = {}

        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_names = [column[1] for column in columns]
            tables_columns[table_name] = column_names

        return tables_columns

def chat_history_customizer(excluded):
    if debugger_print:
        print('history in customizer', history)
    
    cust_history = history    
    for item in cust_history:
        for exc in excluded:
            if len(cust_history) != 0 and exc in list(item['bot'].keys()):
                item['bot'].pop(exc)
    return cust_history






with sqlite3.connect("chinook.db") as con:
    table_info = get_tables_and_columns_sqlite(con)

function_descriptions_multiple = [
    {
        "name": "get_flight_info",
        "description": "Get information of a specific flight",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "db_query",
        "description": f"To make a SQL query and return the results in case of a question about search",
        "parameters": {
            "type": "object",
            "properties": {
                "user_prompt": {
                    "type": "string",
                    "description": "This is the question that user has asked and SQL results will be based on that, e.g. what is the last billing record from 2020",
                },
            },
            "required": ["user_prompt"],
        },
    },
    ]


llm = ChatOpenAI(model=model_name, temperature = 0, openai_api_key = my_api_key)
while True:

    cust_history = chat_history_customizer(['SQL created for user prompt (hidden to user)'])
    
    system_message = f"""You are a data interpreter bot that replies questions by query in database. Consider the conversation history in chat:
    history: {cust_history}
    """
    if debugger_print:
        print('history in main', system_message)

    user_prompt_ = input('ask me: ')

    first_response = llm.predict_messages(
    [HumanMessage(content=user_prompt_),
    SystemMessage(content=system_message),],
    functions = function_descriptions_multiple,
    )

    if len(history) > memory_length:
        history = history[:memory_length]

    try:
        params = first_response.additional_kwargs["function_call"]["arguments"]
        params = params.strip()
        params = json.loads(params)

        chosen_function = eval(first_response.additional_kwargs["function_call"]["name"])
        func_output_1, func_output_2  = chosen_function(**params)

        second_response = llm.predict_messages(
        [
            HumanMessage(content = user_prompt_),
            SystemMessage(content=system_message),
            AIMessage(content = str(first_response.additional_kwargs)),
            AIMessage(
                role = "function",
                additional_kwargs = {
                    "name": first_response.additional_kwargs["function_call"]["name"]
                },
                content = func_output_1,
            ),
        ],
            functions = function_descriptions_multiple,
        )
       
        #if debugger_print:
        print('Response:', second_response.content)
        #print('func out', func_output_1)

        history.insert(0,
        {'Human user': user_prompt_,
        'bot':{
        'bot shown reply': second_response.content,
        'bot chosen function (hidden to user)': first_response.additional_kwargs["function_call"]["name"],
        'SQL created for user prompt (hidden to user)': func_output_2,
        },
        'time': datetime.now()}
        )

    except Exception as e:
        if debugger_print:
            print('error in main', e)
    
        history.insert(0,
        {'Human user': user_prompt_,
        'bot':{
        'bot shown reply': first_response.content,
        'bot chosen function (hidden to user)': None,
        'SQL created for user prompt (hidden to user)': None,
        },
        'Time': datetime.now()}
        )
        if first_response.content:
            print(first_response.content)
        else:
            print(first_response)
