import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from langchain.agents import create_sql_agent, initialize_agent, Tool
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.memory import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory

load_dotenv()

app = Flask(__name__)
port = 4000
CORS(app)


@app.route("/", methods=["POST"])
def chat_completion():
    data = request.json
    message = data["message"]
    userid = data["userid"]

    history = PostgresChatMessageHistory(
        connection_string="postgresql://postgres:admin@localhost:5432/session_db",
        session_id=userid
    )
    memory = ConversationBufferMemory(
        memory_key="history", input_key="input", chat_memory=history, return_messages=True)
    history.add_user_message(message)

    print(f" Memory is: {memory.chat_memory.messages}")
    # print(f" History is: {history}")

    try:
        db = SQLDatabase.from_uri(
            "postgresql://postgres:admin@localhost:5432/sample_db")

        toolkit = SQLDatabaseToolkit(
            db=db, llm=ChatOpenAI(temperature=0))

        SUFFIX = """Begin!
                
        Question: {input}
        Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
        {agent_scratchpad}
        """
        PREFIX = '''You are a SQL expert. You have read only access to a My SQL database.
                    Identify which tables can be used to answer the user's question and write and execute a SQL query accordingly.
                    Do not execute any DDL queries and return Final answer as "DDL query execution not allowed"
                '''
        FORMAT_INSTRUCTIONS = """Please use the following format:
                    
                            '''
                            Thought: 
                            Action: the action to take should be one of [{tool_names}]
                            Action Input: the input to the action
                            Observation: the result of the action
                            '''
                            
                            Provide the output in the Final Answer."""
        agent_executor = create_sql_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            input_variables=["input", "agent_scratchpad"],
            return_intermediate_steps=True,
            top_k=100,
            agent_executor_kwargs={
                'suffix': SUFFIX,
                'prefix': PREFIX,
                'format_instructions': FORMAT_INSTRUCTIONS,
                # "system_message": SystemMessage(content="You are an expert SQL data analyst with read only access"),
                # "memory": memory
            }

        )
        # agent_executor = initialize_agent(
        #     llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        #     tools=toolkit.get_tools(),
        #     verbose=True,
        #     agent=AgentType.OPENAI_FUNCTIONS,
        #     # handle_parsing_errors=True,
        #     # return_intermediate_steps=True,
        #     agent_kwargs={
        #         'prefix': PREFIX,
        #         'format_instructions': FORMAT_INSTRUCTIONS,
        #         'suffix': SUFFIX,
        #         "system_message": SystemMessage(content="You are an expert SQL data analyst with read only access")
        #     }

        # )
        response = agent_executor.run(message)
        print(response)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            history.add_ai_message(e)
            raise e
        response = response.removeprefix(
            "Could not parse LLM output: `").removesuffix("`")
    history.add_ai_message(response)
    return jsonify({"botResponse": response})


@app.route("/history", methods=["POST"])
def get_history():
    data = request.json
    userid = data["userid"]

    history = PostgresChatMessageHistory(
        connection_string="postgresql://postgres:admin@localhost:5432/session_db",
        session_id=userid
    )

    history_data = history.messages

    pairs = [
        {
            "chatPrompt": history_data[i].content,
            "botMessage": history_data[i+1].content,
        }
        # assume the list always start with HumanMessage and then followed by AIMessage
        for i in range(0, len(history_data), 2)
    ]

    return jsonify({"history": pairs})


if __name__ == "__main__":
    app.run(port=port)
