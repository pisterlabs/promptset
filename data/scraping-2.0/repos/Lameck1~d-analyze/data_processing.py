from dateutil.parser import parse
import datetime
import openai
import matplotlib.pyplot as plt
from database import get_db
import io
import re
import os
import base64
import logging


def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')

logging.basicConfig(filename='/tmp/app.log', level=logging.DEBUG)

SAMPLE_SIZE = 100  # Adjustable sample size
SUCCESS_THRESHOLD = 0.5  # Adjustable success rate threshold

# Helper function to check if a value can be cast to a given datatype
def is_type(val: str, datatype: type) -> bool:
    if val is None or val.strip() == '':
        return False
    
    val = val.replace('.', '', val.count('.') - 1).replace(',', '').strip()
    
    try:
        datatype(val)
        return True
    except (ValueError, TypeError):
        return False

# Reusing the helper function
def is_integer(val: str) -> bool:
    val = val.replace('.', '', val.count('.') - 1).replace(',', '').strip()
    if val != '' and is_type(val, int):
        int_val = int(val)
        return int_val >= -2147483648 and int_val <= 2147483647
    return False

def is_float(val: str) -> bool:
    val = val.replace('.', '', val.count('.') - 1).replace(',', '').strip()
    return val != '' and not is_integer(val) and is_type(val, float)

def is_date(string: str, fuzzy=False) -> bool:
    if not string or not isinstance(string, str):
        return False

    try:
        parsed = parse(string, fuzzy=fuzzy)
        # Check if the parsed object is actually a date and the original string contains numbers
        return isinstance(parsed, datetime.date) and any(char.isdigit() for char in string)
    except (ValueError, OverflowError):
        return False


def is_boolean(n: str) -> bool:
    lowered_val = n.strip().lower()
    return lowered_val in ['true', 'false', 'yes', 'no', 'y', 'n', '1', '0']

def format_date(date_str: str) -> str:
    try:
        date_obj = parse(date_str)
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None

# Transformation logic dictionary
TRANSFORMATION_LOGIC = {
    'INTEGER': lambda col, median_value: f"CASE WHEN {col} IS NULL OR TRIM({col}) = '' OR NOT is_integer({col}) THEN {median_value} ELSE CAST(REPLACE({col}, ',', '') AS INTEGER) END",
    'FLOAT': lambda col, mean_value: f"CASE WHEN {col} IS NULL OR TRIM({col}) = '' OR NOT is_float({col}) THEN {mean_value} ELSE CAST(REPLACE({col}, ',', '') AS FLOAT) END",
    'BOOLEAN': lambda col: f"CASE WHEN is_boolean({col}) THEN CAST({col} AS BOOLEAN) ELSE NULL END",
    'DATE': lambda col: f"CASE WHEN is_date({col}, False) THEN format_date({col}) ELSE NULL END",
    'VARCHAR': lambda col: col
}

def infer_column_type(conn, col, relation_name):
    # Retrieve a sample of data
    query = f"SELECT DISTINCT {col} FROM {relation_name} WHERE {col} IS NOT NULL LIMIT ?"
    sample_data = [row[0] for row in conn.execute(query, (SAMPLE_SIZE,)).fetchall()]

    counters = {
        'INTEGER': 0,
        'FLOAT': 0,
        'DATE': 0,
        'BOOLEAN': 0,
        'VARCHAR': 0
    }

    # For each target data type
    for data in sample_data:
        if is_integer(str(data)):
            counters['INTEGER'] += 1
        elif is_float(str(data)):  # Using 'elif' to prioritize integer
            counters['FLOAT'] += 1
        if is_date(str(data)):
            counters['DATE'] += 1
        if is_boolean(str(data)):
            counters['BOOLEAN'] += 1

    # Calculate the success rate for each type
    success_rates = {key: val / len(sample_data) for key, val in counters.items()}

    # Assign the column type based on success rates
    max_rate = max(success_rates.values())
    if max_rate < SUCCESS_THRESHOLD:
        return 'VARCHAR'
    else:
        return max([key for key in success_rates if success_rates[key] == max_rate])

def transform_cast(col: str, type: str) -> str:
    return TRANSFORMATION_LOGIC.get(type, TRANSFORMATION_LOGIC['DEFAULT'])(col)

def table_exists(conn, table_name):
    try:
        conn.execute(f"SELECT * FROM {table_name} LIMIT 1")
        return True
    except:
        return False

def get_columns(conn, table_name):
    result = conn.execute(f"SELECT * FROM {table_name} LIMIT 1")
    return [desc[0] for desc in result.description]

def infer_and_set_datatypes(relation_name):
    conn = get_db()

    # register the helper functions with DuckDB
    conn.create_function("is_integer", is_integer)
    conn.create_function("is_float", is_float)
    conn.create_function("is_boolean", is_boolean)
    conn.create_function("is_date", is_date)
    conn.create_function("format_date", format_date)
    conn.create_function("transform_cast", transform_cast)

    # Check table existence
    if not table_exists(conn, relation_name):
        raise ValueError(f"Table {relation_name} does not exist.")

    # Retrieve the column names in the relation
    columns = get_columns(conn, relation_name)

    inferred_columns = []

    # Iterate through columns and perform type inference
    for col in columns:
        inferred_type = infer_column_type(conn, col, relation_name)
        inferred_columns.append((col, inferred_type))

        # If the inferred type is INTEGER or FLOAT, compute the mean or median
        if inferred_type in ['INTEGER', 'FLOAT']:
            value_to_replace = compute_mean_or_median(conn, col, relation_name, inferred_type)
            cast_expression = TRANSFORMATION_LOGIC[inferred_type](col, value_to_replace)
        else:
            cast_expression = TRANSFORMATION_LOGIC[inferred_type](col)

        # Add new column without setting its values
        try:
            conn.execute(f"ALTER TABLE {relation_name} ADD COLUMN {col}_new {inferred_type};")
        except Exception as e:
            print(f"Error when adding new column '{col}_new': {e}")

        # Update new column with transformed values
        try:
            conn.execute(f"UPDATE {relation_name} SET {col}_new = {cast_expression};")
        except Exception as e:
            print(f"Error when processing column '{col}': {e}")

        # Drop original column and rename new column
        try:
            conn.execute(f"ALTER TABLE {relation_name} DROP COLUMN {col};")
            conn.execute(f"ALTER TABLE {relation_name} RENAME COLUMN {col}_new TO {col};")
        except Exception as e:
            print(f"Error when renaming column '{col}': {e}")

    return relation_name, inferred_columns

def compute_mean_or_median(conn, col, relation_name, datatype):
    if datatype == 'FLOAT':
        query = f"SELECT AVG(CAST(REPLACE({col}, ',', '') AS FLOAT)) FROM {relation_name} WHERE is_float({col})"
    elif datatype == 'INTEGER':
        query = f"SELECT MEDIAN(CAST(REPLACE({col}, ',', '') AS INTEGER)) FROM {relation_name} WHERE is_integer({col})"
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")
    
    result = conn.execute(query).fetchone()
    return result[0] if result else None


def get_sql_query(user_query, column_names, upload_id):
    openai.api_key = get_openai_api_key()
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are a SQL generator. Strictly use only the available columns: {column_names}, table name is {upload_id}. Please respond with a SQL query enclosed between ### symbols (Example: ###query###). If you can't generate an sql query for question, explain user why this question not answerable\n\nWhat is your question about this dataset?\n\n{user_query}",
        max_tokens=200
    )
    sql_query = completion.choices[0].text.strip()
    cleaned_sql_query = re.search("###(.*?)###", sql_query, re.DOTALL)
    return cleaned_sql_query.group(1).strip() if cleaned_sql_query else None


def get_assistant_response(user_query, cleaned_sql_query, query_results):
    openai.api_key = get_openai_api_key()
    prompt_content = f"You are a data analysis system. The user asked the following question: '{user_query}'. You performed the following DuckDB query to find the answer: '{cleaned_sql_query}'. The results are: {query_results}. Please respond with the answer to the user's question."
    response_completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_content,
        max_tokens=150
    )
    return response_completion.choices[0].text.strip()


def generate_charts(query_results):
    try:
        if len(query_results[0]) == 2:
            labels, data = zip(*query_results)

            if isinstance(data[0], (int, float)) and isinstance(labels[0], (int, float)):
                # For numeric labels and data, create all four charts
                fig, ax = plt.subplots(2, 2, figsize=(14,10))

                # Bar plot
                ax[0, 0].bar(labels, data)
                ax[0, 0].set_xticks(range(len(labels)))  # Updated this line
                ax[0, 0].set_xticklabels(labels, rotation=45, horizontalalignment='right')  # And this line
                ax[0, 0].set_title("Bar plot")

                # Line plot
                ax[0, 1].plot(labels, data)
                ax[0, 1].set_title("Line plot")

                # Scatter plot
                ax[1, 0].scatter(labels, data)
                ax[1, 0].set_title("Scatter plot")

                # Box plot
                ax[1, 1].boxplot(data, vert=False)
                ax[1, 1].set_title("Box plot")

            else:
                # For categorical labels and numeric data, only create a bar plot
                fig, ax = plt.subplots(figsize=(10,6))  # Adjust the figure size
                ax.bar(labels, data)
                ax.set_xticks(range(len(labels)))  # Updated this line
                ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')  # And this line
                ax.set_title("Bar plot")

        elif len(query_results[0]) == 1 and isinstance(query_results[0][0], (int, float)):
            data, = zip(*query_results)

            fig, ax = plt.subplots(1, 2, figsize=(14,5))

            # Histogram
            ax[0].hist(data, bins=10)
            ax[0].set_title("Histogram")

            # Box plot
            ax[1].boxplot(data, vert=False)
            ax[1].set_title("Box plot")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Could not generate charts: {str(e)}")
        return None