from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import psycopg2
from psycopg2 import pool
import uuid
import bcrypt
import json
import time
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor, load_tools
import os
from langchain import SQLDatabase, SQLDatabaseChain, OpenAI
from langchain.chains import LLMChain
import yaml
from langchain.requests import RequestsWrapper
from langchain.tools.json.tool import JsonSpec
from langchain.chat_models import ChatOpenAI
from langchain.chains import APIChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.utilities import SerpAPIWrapper
from decouple import config

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
os.environ["SERPAPI_API_KEY"] = config('SERPAPI_API_KEY')

search = SerpAPIWrapper()


def get_hashed_password(plain_text_password):
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    return bcrypt.checkpw(plain_text_password, hashed_password)


host = config('host')
dbname = "postgres"
user = config('user')
password = config('password')
sslmode = "require"

conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(
    host, user, dbname, password, sslmode)

postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(1, 20, conn_string)
if (postgreSQL_pool):
    print("Connection pool created successfully")

conn = postgreSQL_pool.getconn()

cursor = conn.cursor()


@ app.route('/')
def hello_world():
    return 'Hello, World!'


@ app.route('/signup', methods=['POST'])
@ cross_origin()
def signup():
    id = uuid.uuid1()
    temp_json = json.loads(request.data)
    name = temp_json['name']
    email = temp_json['email']
    print(email)
    token = get_hashed_password(password)
    age = temp_json['age']
    weight = temp_json['weight']
    sex = temp_json['sex']
    height = temp_json['height']
    body_fat = temp_json['body_fat']
    cursor.execute(
        f"INSERT INTO users (id, name, email, age, sex, weight, height, body_fat, token) values ('{id}', '{name}', '{email}', '{age}', '{sex}' , '{weight}', '{height}', '{body_fat}', '{token}');")
    conn.commit()
    return token


@ app.route('/signin', methods=['POST'])
def signin():
    temp_json = json.loads(request.data)
    email = temp_json['email']
    password = temp_json['password']
    cursor.execute(f"SELECT token FROM users WHERE email = '{email}';")
    conn.commit()
    token = cursor.fetchone()[0]
    if (check_password(password, token)):
        print("auth_success")
    else:
        print("auth_unsuccess")
    return token


@ app.route('/getconvos', methods=['POST'])
def getconvos():
    temp_json = json.loads(request.data)
    token = temp_json['token']
    cursor.execute(f"SELECT id FROM users WHERE token = '{token}';")
    print(token)
    fetchconvos = cursor.fetchone()
    print(fetchconvos)
    if fetchconvos is not None:
        id = fetchconvos[0]
        cursor.execute(
            f"SELECT conversation FROM conversations WHERE user_id = '{id}';")
        convo = cursor.fetchone()[0]
        conn.commit()
        return json.dumps(convo)


@ app.route('/sendmessage', methods=['POST'])
def sendmessage():
    temp_json = json.loads(request.data)

    message = temp_json['message']
    token = temp_json['token']

    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm1 = OpenAI(temperature=0)

    db = SQLDatabase.from_uri(
        """postgresql://postgres:TnZe*sJY3OMe|"%<@34.148.191.128""")

    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    nutrinix_headers = {
        "x-app-id": "13509888",
        "x-app-key": "dbacf5040facc8cbfa01ebc89cf3ed60",
        "x-remote-user-id": 0
    }

    nutrinix_api = """
    Base Url: https://trackapi.nutritionix.com

    Endpoints:

    POST /v2/natural/nutrients
    Summary: Get detailed nutrient breakdown of any natural language text.  Can also be used in combination with the /search/instant endpoint to provide nutrition information for common foods.
    Body Parameters Example: { "query": "string", "num_servings": 0, "aggregate": "string", "line_delimited": false, "use_raw_foods": false, "include_subrecipe": false, "timezone": "string", "consumed_at": "string","lat": 0, "lng": 0, "meal_type": 0, "use_branded_foods": false, "locale": "string", "taxonomy": false, "ingredient_statement": false, "last_modified": false
    }
    """

    nutrition_chain = APIChain.from_llm_and_api_docs(
        llm, nutrinix_api, headers=nutrinix_headers, verbose=True)

    tools = [
        Tool(
            name="SQL Database",
            func=db_chain.run,
            description="""Runs a query on an SQL Database. You are able to query data, modify data, or insert data. The following tables are in the db:

                users:
                    -id: uuid
                    -name: text
                    -email: text
                    -age:int4
                    -weight: float4
                    -height: text
                    -body fat: float4
                    -token: text

                    goals:
                    -id: uuid
                    -user_id: uuid
                    -goal weight: float4
                    -goal body fat: float4

                    workout_split:
                    -id: uuid
                    -user_id: uuid
                    -detailed_description: text

                    workouts:
                    -id: uuid
                    -user_id: uuid
                    -summary: text
                    -exercise: text
                    -calories burned: int4
                    -time: timestamp

                    meals:
                    -id: uuid
                    -user_id: uuid
                    -meal_name: text
                    -ingredients: text
                    -calories: int4
                    -fat: int4
                    -protein: int4
                    -carbs: int4
                    -meal_type: text
                    -time: timestamp

                    PRs:
                    -id: uuid
                    -user_id: uuid
                    -exercise: text
                    -weight: float4
                    -time: timestamp
                """
        ),
        Tool(
            name="Google Search",
            func=search.run,
            description="Can be used to get advice, nutritional information, workouts, or anything else. Input should be a query."
        ),
        # Tool(
        #     name="Nutrition Information",
        #     func=nutrition_chain.run,
        #     description="Can retrieve nutritional information from a natural language query."
        # )
    ]

    cursor.execute(f"SELECT id FROM users WHERE token = '{token}';")
    id = cursor.fetchone()[0]
    cursor.execute(
        f"SELECT conversation FROM conversations WHERE user_id = '{id}';")
    fetchconvos = cursor.fetchone()
    # print(fetchconvos)
    temp_memory = ""
    if fetchconvos is None:
        conversation = json.dumps(
            {'messages': [{'message': message, 'type': 'user', 'timestamp': time.time()}]})
        cursor.execute(
            f"INSERT INTO conversations (id, user_id, conversation) values ('{uuid.uuid1()}', '{id}', '{conversation}');")
    else:
        convo = fetchconvos[0]
        convo_json = json.loads(convo)
        convo_json['messages'].append(
            {"message": message, "type": "user", 'timestamp': time.time()})
        cursor.execute(
            f"UPDATE conversations SET conversation = '{json.dumps(convo_json)}' WHERE user_id = '{id}';")
        temp_memory = convo

    agent_memory = ""

    # if temp_memory != "":
    #     json.loads(temp_memory)[0]

    # print(temp_memory)
    # print(temp_memory['messages'])

    prefix = """You are an AI personal trainer named Muscle Memory! You have access to the following tools:"""
    suffix = """Begin! The user which you are talking to has an id of {}. If you need to create a new row in the db, use this id: {}. Don't feel the need to give the user any id's, just give them the information requested or tell them their request was completed.

    YOU CAN INSERT DATA INTO AN SQL DATABASE!!
    """.format(id, uuid.uuid1())

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=[]
    )

    messages = [
        SystemMessagePromptTemplate(prompt=prompt),
        HumanMessagePromptTemplate.from_template("{input}\n\nThis was your previous work "
                                                 f"(but I haven't seen any of it! I only see what "
                                                 "you return as final answer):\n{agent_scratchpad}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    llm_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, )
    return agent_executor.run(message)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
