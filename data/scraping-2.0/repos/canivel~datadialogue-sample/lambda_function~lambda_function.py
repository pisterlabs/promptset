import json
import os
import boto3
from sqlalchemy.engine import create_engine
from langchain_experimental.sql import SQLDatabaseChain
from langchain import PromptTemplate, SQLDatabase, LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Environment variables
GLUE_DB_NAME = os.environ["GLUE_DB_NAME"]
ATHENA_S3_RESULTS_BUCKET = os.environ["ATHENA_S3_RESULTS_BUCKET"]
OPENAI_SECRET_ID = os.environ["OPENAI_API_SECRET_NAME"]

# Set up OpenAI Chat Model
client = boto3.client("secretsmanager")
OPENAI_API_KEY = client.get_secret_value(SecretId=OPENAI_SECRET_ID)["SecretString"]
chat_model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")

# Set up Athena connection
region = client.meta.region_name
athena_connection_params = {
    "conn": f"athena.{region}.amazonaws.com",
    "port": "443",
    "schema": GLUE_DB_NAME,
    "s3_staging": f"s3://{ATHENA_S3_RESULTS_BUCKET}/athenaresults/",
    "work_group": "primary"
}
connection_string = (
    f"awsathena+rest://@{athena_connection_params['conn']}:{athena_connection_params['port']}/"
    f"{athena_connection_params['schema']}?s3_staging_dir={athena_connection_params['s3_staging']}/"
    f"&work_group={athena_connection_params['work_group']}"
)
athena_engine = create_engine(connection_string, echo=False)
athena_db = SQLDatabase(athena_engine)


def fetch_glue_catalog():
    """
    Fetches the AWS Glue catalog and returns a formatted string representation.
    """
    columns_str = ""
    glue_client = boto3.client("glue")
    
    for db in [GLUE_DB_NAME]:
        tables_response = glue_client.get_tables(DatabaseName=db)
        for table in tables_response["TableList"]:
            if table["StorageDescriptor"]["Location"].startswith("s3"):
                classification = "s3"
            else:
                classification = table["Parameters"]["classification"]
            
            for column in table["StorageDescriptor"]["Columns"]:
                columns_str += f"\n{classification}|{table['DatabaseName']}|{table['Name']}|{column['Name']}"
    
    return columns_str


def determine_data_channel(query):
    """
    Uses the chat model to determine the appropriate database and table for a given query.
    """
    glue_catalog_str = fetch_glue_catalog()
    prompt_template = (
        f"""
        From the table below, find the database (in column database) which will contain the data
        (in corresponding column_names) to answer the question {query} \n{glue_catalog_str}
        Give your answer as database == 
        Also, give your answer as database.table == 
        """
    )
    
    channel_prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    llm_chain = LLMChain(prompt=channel_prompt, llm=chat_model)
    generated_texts = llm_chain.run(query)
    
    return "db", athena_db


def execute_query(query):
    """
    Uses the chat model to generate an SQL query and then executes it.
    """
    channel, db = determine_data_channel(query)
    
    sql_prompt_template = """
    Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Do not append 'Query:' to SQLQuery.
    Do not use Backquotes in SQL queries, please use double quotes when needed.
    Display SQLResult after the query is run in plain English that users can understand.
    Provide answer in simple English statement.
    Only use the following tables:
    {table_info}
    Question: {input}
    """
    
    sql_prompt = PromptTemplate(
        input_variables=["input", "table_info", "dialect"],
        template=sql_prompt_template
    )
    
    db_chain = SQLDatabaseChain.from_llm(
        chat_model, db, prompt=sql_prompt, verbose=True, return_intermediate_steps=False
    )
    
    return db_chain.run(query)


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    """
    body = json.loads(event["body"])
    query = body.get("query", "")
    
    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"message": "Query not provided"}),
            "headers": {"Content-Type": "application/json"},
        }
    
    try:
        response_data = execute_query(query)
    except Exception as e:
        modified_query = f"{query} When querying please consider the previous exception: {str(e)[:500]}"
        response_data = execute_query(modified_query)
    
    return {
        "statusCode": 200,
        "body": json.dumps(response_data),
        "headers": {"Content-Type": "application/json"},
    }
