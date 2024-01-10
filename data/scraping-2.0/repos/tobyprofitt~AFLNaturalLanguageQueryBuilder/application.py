from flask import Flask, request, jsonify, render_template
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from dotenv import load_dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)

TEMPLATE = """You are Anthony (Huddo) Hudson, an AFL commentator and stat expert. You also know SQL and have access to an SQLite3 database of AFL statistics.
Given an input question, first create a syntactically correct SQLite3 query to run, then look at the results of the query and return the answer.

Only use the following tables:

player_stats and game.

Few shot examples:
Question: "How many goals did Zak Butters kick in 2023?"
SQLQuery: "SELECT SUM(goals) FROM player_stats WHERE player_name = 'Zak Butters' AND year = 2023"

Question: "How many goals did Max Gawn kick last year?"
SQLQuery = "SELECT SUM(goals) FROM player_stats WHERE player_name = 'Max Gawn' AND year = strftime('%Y', date('now', '-1 year'))"

Whenever you are asked a question, please return additional information that might be implied by the question, such as what year or round a statistic is from.

If you get no result, or an empty table, from the query then return "No result" and suggest a new query to run.

Question: {user_question}"""

load_dotenv()  # This loads the .env variables into the environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

application = Flask(__name__)

# Setup LangChain and SQLite
db = SQLDatabase.from_uri("sqlite:///afl.db")
llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

'''Capture Print'''
import io
import sys
from contextlib import redirect_stdout

def capture_print(func):
    def wrapper(*args, **kwargs):
        # Create a new StringIO object
        temp_out = io.StringIO()

        # Redirect stdout to the StringIO object
        with redirect_stdout(temp_out):
            result = func(*args, **kwargs)

        # Get the contents of the StringIO object and then restore stdout
        captured_output = temp_out.getvalue()

        # Return both the function result and the captured output
        return result, captured_output
    return wrapper

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/query', methods=['POST'])
def query():
    try:
        logger = logging.getLogger("query_route")
        data = request.json
        logger.info(f"Data received: {data}")

        question = data['question']
        logger.info(f"Question asked: {question}")

        @capture_print
        def query_internal():
            return db_chain.run(TEMPLATE.format(user_question=question))

        result, captured_output = query_internal()
        logger.info("Captured output from db_chain.run: %s", captured_output)
        logger.info(f"Result: {result}")

        return jsonify(result)
    except Exception as e:
        return jsonify("Sorry, I couldn't quite figure that one out. Try rephrasing your question, or come back later.")

if __name__ == '__main__':
    application.run(debug=True)
