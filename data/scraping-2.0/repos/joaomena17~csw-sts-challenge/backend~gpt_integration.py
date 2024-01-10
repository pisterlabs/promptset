import openai
import sqlite3
import os

openai.api_key = os.getenv("OPEN_AI_KEY")

sensors_columns = [
    "id",
	"name",
	"type",
	"office",
	"building",
	"room",
	"units" 
]

sensor_values_columns = [
    "sensor",
	"timestamp",
	"value"
]

database_structure = """
                    CREATE TABLE IF NOT EXISTS "sensors" (
                        "id" INTEGER PRIMARY KEY,
                        "name" TEXT COLLATE NOCASE,
                        "type" TEXT COLLATE NOCASE,
                        "office" TEXT COLLATE NOCASE,
                        "building" TEXT COLLATE NOCASE,
                        "room" TEXT COLLATE NOCASE,
                        "units" TEXT COLLATE NOCASE
                    )

                    CREATE TABLE IF NOT EXISTS "sensor_values" (
                        "sensor" INTEGER,
                        "timestamp" TEXT,
                        "value" REAL
                    )  
                    """

def generate_sql_query(text, table1= "sensors", table2 = "sensor_values", columns1 = sensors_columns, columns2 = sensor_values_columns):
    prompt = """You are a language model that can generate SQL queries. \
                Please provide a natural language input text, \
                and I will generate the corresponding SQL query for you. \
                which will be compatible with SQLite. \
                The table names are {} nad {} and the correstounding \
                columns are {} and {} .\nInput: {}\nSQL Query:""".format(
                    table1, table2, columns1, columns2, text
                )
    
    request = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0301",
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    sql_query = request["choices"][0]["message"]["content"]

    return sql_query

def virify_sql_query(query, structure = database_structure):
    prompt = f"""As a language model, I am capable of verifying SQL queries. Please provide an input query, and I will check if
                this query represents a potential threat to our database. We only allow SELECT queries
                and any modifications to the data base are prohibited. The structure of out database is as follows: {structure}. \nSQL Query: {query}" \n
                If the query is safe and complies with the rules, I will return "True". If it is potential
                harmful or not compliant, I will return "False".
                """
    
    request = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0301",
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    response = request["choices"][0]["message"]["content"]

    return response
