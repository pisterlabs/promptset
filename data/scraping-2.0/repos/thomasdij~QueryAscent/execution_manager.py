import openai
import logging
import pandas as pd
from global_event_publisher import event_publisher
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from utils import truncate_content

class SQLExecutionError(Exception):
    """Custom exception for SQL execution errors."""
    def __init__(self, original_exception):
        self.original_exception = str(original_exception)
        super().__init__(self.original_exception)

def prompt_on_df(prompt_directive, df, model="gpt-3.5-turbo"):
    prompt = (prompt_directive + df.to_string(index=True))
    completion = openai.ChatCompletion.create(
    model=model,
    temperature=0,
    messages=[{"role": "user", "content": prompt}]
    )
    logging.debug(f"PROMPT: {truncate_content(prompt_directive)}")
    logging.debug(f"RESPONSE: {truncate_content(completion.choices[0].message['content'])}\n")
    return completion.choices[0].message['content']

def prompt_on_directive(messages, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        messages=messages
    )
    logging.debug(f"PROMPT: {truncate_content(messages)}")
    logging.debug(f"RESPONSE: {truncate_content(completion.choices[0].message['content'])}\n")
    return completion.choices[0].message['content']

def run_sql(sql: str, database_url: str) -> pd.DataFrame:
    logging.debug("About to try executing SQL...")  
    
    try:
        logging.debug("Creating engine...")  
        engine = create_engine(database_url)
        logging.debug(f"Engine created: {engine}")

        logging.debug("Executing SQL query...")  
        sql_answer_df = pd.read_sql_query(sql, engine)
        return sql_answer_df
    
    except SQLAlchemyError as e:
        logging.debug("Type of original SQLAlchemy exception:", type(e.orig))
        logging.debug("Attributes of original SQLAlchemy exception:", dir(e.orig))
        logging.debug(e.orig)
        raise SQLExecutionError(e.orig)    
    
    except Exception as e:  # Temporarily catch all exceptions
        logging.debug("An exception occurred!")  
        logging.debug(e)
        # Re-raise as a custom exception
        raise SQLExecutionError(e)

def generate_sql_query(question, filtered_schema_and_synonyms_df):
    prompt_directive = (
        "I provide a question and a Database Schema Table and you provide SQL."
        "The Database Schema Table is meta-information: each row represents a column in a specific table within the "
        "database. It details the 'table_name', 'column_name', 'data_type', and, if applicable, 'synonym_list' for that column."
        "You will only respond with SQL code and not with any explanations."
        "Here is the question: " +
        question +
        "Here is the Database Schema Table:\n"
    )
    sql_query = prompt_on_df(prompt_directive, filtered_schema_and_synonyms_df)
    event_publisher.emit("initial_sql_query_set", sql_query)
    return sql_query

def execute_sql_with_fallback(sql_query, database_url, question, filtered_schema_and_synonyms_df):
    
    conversation = []

    def execute_sql_with_fallback_subcall(sql_query, database_url, question, filtered_schema_and_synonyms_df, conversation = None, retry_count=0, last_sql_query=None):
        try:
            event_publisher.emit(f"fallback_query_{retry_count}_set", sql_query)
            sql_result = run_sql(sql_query, database_url)
            return sql_result
        except SQLExecutionError as e:
            event_publisher.emit(f"fallback_exception_{retry_count}_set", e.original_exception)
            if sql_query == last_sql_query:
                logging.debug("SQL query is the same for two loops in a row. Switching to error analysis.")
                return execute_sql_with_error_analysis(sql_query, database_url, question, filtered_schema_and_synonyms_df, e.original_exception)
            
            if retry_count < 5:
                logging.debug(f"An error occurred while executing SQL: {e.original_exception}. Retrying... ({retry_count + 1}/5)\n")
                try:
                    # Assuming `attempt_to_fix_sql_query` returns a modified SQL query
                    conversation = attempt_to_fix_sql_query(sql_query, question, e.original_exception, filtered_schema_and_synonyms_df, conversation, retry_count)
                    fixed_sql_query = conversation[-1]["content"]

                    # Recursive call with incremented retry_count and last sql query updated
                    return execute_sql_with_fallback_subcall(fixed_sql_query, database_url, question, filtered_schema_and_synonyms_df, conversation, retry_count + 1, last_sql_query=sql_query)
                    
                except Exception as fix_e:
                    logging.debug(f"An error occurred while attempting to fix the SQL: {fix_e}. Retrying... ({retry_count + 1}/5)\n")
                    fixed_sql_query = conversation[-1]["content"]
                    return execute_sql_with_fallback_subcall(fixed_sql_query, database_url, question, filtered_schema_and_synonyms_df, conversation, retry_count + 1, last_sql_query=sql_query)
            else:
                logging.debug(f"Exceeded maximum number of retries. Latest error: {e.original_exception}. Switching to error analysis.")
                return execute_sql_with_error_analysis(sql_query, database_url, question, filtered_schema_and_synonyms_df, e.original_exception)

    return execute_sql_with_fallback_subcall(sql_query, database_url, question, filtered_schema_and_synonyms_df, conversation, retry_count=0, last_sql_query=None)

def attempt_to_fix_sql_query(sql_query, question, exception, reference_df, conversation, retry_count):
    if retry_count == 0:
        prompt_directive = (
            "I provide you with an SQL query that has produced an error, along with the original question it's meant to answer, "
            "the specific error message, and a Database Schema Table. The Database Schema Table is meta-information: each row "
            "represents a column in a specific table within the database. It details the 'table_name', 'column_name', "
            "'data_type', and, if applicable, 'synonym_list' for that column. Your task is to diagnose the SQL error based on the "
            "given information and produce a corrected query that should successfully execute and be different from the original one. "
            "Please only respond with the corrected SQL code, without any additional explanations.\n"
            "Here is the original question the query is meant to answer: " +
            question +
            "\nHere is the original query:\n" +
            sql_query +
            "\nHere is the error message:\n" +
            exception +
            "\nHere is the Database Schema Table:\n" +
            reference_df.to_string()
        )
        conversation.append({"role": "user", "content": prompt_directive})

        prompt = (prompt_directive)
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
        )
        logging.debug(f"PROMPT: {truncate_content(prompt_directive)}")
        logging.debug(f"RESPONSE: {truncate_content(completion.choices[0].message['content'])}\n")
        fixed_sql_result = completion.choices[0].message['content']

        conversation.append({"role": "assistant", "content": fixed_sql_result})
        logging.debug(f"Original SQL query: {sql_query}\n")
        logging.debug(f"New SQL query: {fixed_sql_result}\n")
    else:
        prompt_directive = (
            "The SQL query you supplied returned the following error message:\n" + exception +
            "\nPlease provide a new, corrected SQL query that should successfully execute and is "
            "different from the original. Your response must be different from the original query "
            "and it must answer the user's original question."
            "Please only respond with the corrected SQL code, without additional explanations."
        )
        conversation.append({"role": "user", "content": prompt_directive})
        fixed_sql_result = prompt_on_directive(conversation)
        conversation.append({"role": "assistant", "content": fixed_sql_result})
        logging.debug(f"New SQL query: {fixed_sql_result}\n")
    return conversation

def execute_sql_with_error_analysis(sql_query, database_url, question, filtered_schema_and_synonyms_df, exception):
    conversation = []

    # Set Direction and Summarize the User's Original Question
    directive1 = (
        "I provide you with an SQL query that has produced an error, along with the original question it's meant to answer,"
        "the specific error message, and a Database Schema Table. Your task is to help diagnose and correct the issue. "
        "Please start by summarizing the user's original question: " + question
                )
    conversation.append({"role": "user", "content": directive1})
    summary_response = prompt_on_directive(conversation)
    conversation.append({"role": "assistant", "content": summary_response})
    event_publisher.emit("error_analysis_content_1_set", summary_response)

    # Original SQL Query Intent
    directive2 = "What do you think the original SQL query is trying to do?\nSQL Query: " + sql_query
    conversation.append({"role": "user", "content": directive2})
    intent_response = prompt_on_directive(conversation)
    conversation.append({"role": "assistant", "content": intent_response})
    event_publisher.emit("error_analysis_content_2_set", intent_response)

    # Error Message Interpretation
    directive3 = "What do you make of the following error message?\nError Message: " + exception
    conversation.append({"role": "user", "content": directive3})
    error_interpretation = prompt_on_directive(conversation)
    conversation.append({"role": "assistant", "content": error_interpretation})
    event_publisher.emit("error_analysis_content_3_set", error_interpretation)

    # Schema-based Revision Needs
    directive4 = (
        "Based on this Database Schema, what changes would you suggest for the original query? The Database Schema Table is"
        "meta-information: each row represents a column in a specific table within the database. It details the 'table_name',"
        " 'column_name', 'data_type', and, if applicable, 'synonym_list' for that column. \nDatabase Schema:\n" + 
        filtered_schema_and_synonyms_df.to_string()
            )
    conversation.append({"role": "user", "content": directive4})
    schema_revision_suggestion = prompt_on_directive(conversation)
    conversation.append({"role": "assistant", "content": schema_revision_suggestion})
    event_publisher.emit("error_analysis_content_4_set", schema_revision_suggestion)

    # Steps to Fix SQL Query
    directive5 = "What are the steps to correct the SQL query based on the above information?"
    conversation.append({"role": "user", "content": directive5})
    steps_to_fix = prompt_on_directive(conversation)
    conversation.append({"role": "assistant", "content": steps_to_fix})
    event_publisher.emit("error_analysis_content_5_set", steps_to_fix)

    # Corrected SQL Query
    directive6 = (
        "Based on all the information and suggestions, please provide a corrected SQL query that should successfully execute. "
        "Please only respond with the corrected SQL code, without additional explanations. "
        "For example, do not preface your response with 'The corrected SQL query is...'\n"
        "\nOriginal Question:"  + question + "\nOriginal SQL Query: " + sql_query + "\nError Message: " + exception + 
        "\nDatabase Schema:\n" + filtered_schema_and_synonyms_df.to_string()      
                )  
    conversation.append({"role": "user", "content": directive6})
    corrected_sql_query = prompt_on_directive(conversation)
    event_publisher.emit("error_analysis_content_6_set", corrected_sql_query)
    try:
        sql_result = run_sql(corrected_sql_query, database_url)
        return sql_result
    except SQLExecutionError as e:
        logging.debug("Corrected SQL query still produces an error. Quitting.")
        return "Unable to answer."
