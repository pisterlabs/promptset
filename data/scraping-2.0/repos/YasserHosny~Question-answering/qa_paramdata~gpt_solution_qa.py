import openai
import pandas as pd
import json
from dotenv import load_dotenv
import time
from db_data_handle import load_data_from_db_VIEW_DATA_PART, load_data_from_db_se_flat_data, exec_query_over_db, parse_sql_query, load_data_from_elastic, exec_elasticsearch_search
import re
from gpt_key_handle import get_gpt_key

load_dotenv()
gpt_key_index = 1
print("gpt_key_index: ", gpt_key_index)
openai.api_key = get_gpt_key(gpt_key_index)
elastic_colmns = load_data_from_elastic()


def apply_gpt_sql_query(query):
    jsonRows, csvRows, sql_columns = load_data_from_db_VIEW_DATA_PART(1)
    prompt = """Ignore any previous conversations. Please regard the following table columns: {}

    The table name is VIEW_DATA_PART. Use ' as the quote character. Quote column aliases with ". Write a MS SQL Server query to answer the following question: 
    ''
    {}
    ''
    """.format(sql_columns, query)
    print(prompt)

    sql_query = getResultFromChatGPT(prompt)

    print("===> {}: {}\n".format(query, sql_query))

    try:
        rows, cols = exec_query_over_db(sql_query)
    except Exception as e:
        # Handle the exception
        print(f"An error occurred(gpt_solution_qa): {str(e)}")
        rows = []
        cols = []
        # Retry get query from ChatGPT
        sql_query = getResultFromChatGPT(prompt)
        rows, cols = exec_query_over_db(sql_query)

    return rows, cols, sql_query


def apply_gpt_sql_elastic_query(user_query, dataset_name):
    if (dataset_name == "VIEW_DATA_PART"):
        jsonRows, csvRows, sql_columns = load_data_from_db_VIEW_DATA_PART(1)
        # unit_handle = "Only if MS SQL Query WHERE clause has 'Feature Unit' field set fuzziness of 2 for it"
    elif (dataset_name == "se_flat_data"):
        jsonRows, csvRows, sql_columns = load_data_from_db_se_flat_data(1)

    prompt1 = """Ignore any previous conversations. Please regard the following table columns: {}

    The table name is {}. Use ' as the quote character. Quote column aliases with ". Write a MS SQL Server query to answer the following question: 
    ''
    {}
    ''

    Then convert your generated MS SQL Server query to an elastic search query and follow the following instructions sharply:
    1. Only if the MS SQL Server query has 'Feature Unit' field in WHERE clause, set fuzziness of 2 for it.
    2. If the MS SQL Server query does not have 'Feature Unit' field in WHERE clause, DON'T use 'Feature Unit'.
    3. Ignore using 'sort' operation in Elastic Search query, even if the MS SQL Server query has order by clause.
    """.format(sql_columns, dataset_name, user_query)

    elastic_query = getResultFromChatGPT(prompt1)
    print("SQL Query===> {}: {}\n".format(prompt1, elastic_query))

    # prompt2 = """Ignore any previous conversations. Convert the following MS SQL Server query to an elastic search query: {}
    # Follow the following instructions sharply:
    # 1. Only if the MS SQL Server query has 'Feature Unit' field in WHERE clause, set fuzziness of 2 for it.
    # 2. If the MS SQL Server query does not have 'Feature Unit' field in WHERE clause, DON'T use 'Feature Unit'.
    # 3. Ignore using 'sort' operation in Elastic Search query, even if the MS SQL Server query has order by clause.
    # """.format(sql_query)

    # elastic_query = getQueryFromChatGPT(prompt2)

    # print("Elastic Query===> {}: {}\n".format(prompt2, elastic_query))

    try:
        rows, cols = exec_elasticsearch_search(elastic_query)
        # Tries count = 3
        try_count = 3
        if len(rows) == 0:
            while try_count > 0:
                if len(rows) == 0:
                    # sleep for 3 second
                    time.sleep(3)
                    elastic_query = getResultFromChatGPT(prompt1)
                    print("Elastic Query try{}===> {}: {}\n".format(
                        try_count, prompt1, elastic_query))
                    rows, cols = exec_elasticsearch_search(elastic_query)
                try_count -= 1
    except Exception as e:
        # Handle the exception
        print(f"An error occurred(gpt_solution_qa): {str(e)}")
        rows = []
        cols = []
        # Retry get query from ChatGPT
        elastic_query = getResultFromChatGPT(prompt1)
        rows, cols = exec_elasticsearch_search(elastic_query)

    print("rows: {}, cols: {}\n".format(rows, cols))
    return rows, cols, '', elastic_query


def apply_gpt_sql_elastic_translate_query(user_question, dataset_name):
    if (dataset_name == "VIEW_DATA_PART"):
        jsonRows, csvRows, sql_columns = load_data_from_db_VIEW_DATA_PART(1)
        # unit_handle = "Only if MS SQL Query WHERE clause has 'Feature Unit' field set fuzziness of 2 for it"
    elif (dataset_name == "se_flat_data"):
        jsonRows, csvRows, sql_columns = load_data_from_db_se_flat_data(1)

    prompt = """Ignore any previous conversations. Please regard the following table columns: {}

    The table name is {}. Use ' as the quote character. Quote column aliases with ". Write a MS SQL Server query to answer the following question: 
    ''
    {}
    ''

    Then convert your generated MS SQL Server query to an elastic search query and follow the following instructions sharply:
    1. Only if the MS SQL Server query has 'Feature Unit' field in WHERE clause, set fuzziness of 2 for it.
    2. If the MS SQL Server query does not have 'Feature Unit' field in WHERE clause, DON'T use 'Feature Unit'.
    3. Ignore using 'sort' operation in Elastic Search query, even if the MS SQL Server query has order by clause.

    Then detect the language of the following question:
    {}

    Output the results in the following format:
    1. SQL Server query:
    2. Elastic Search query:
    3. Detected language:
    """.format(sql_columns, dataset_name, user_question, user_question)
    print("prompt: {}\n".format(prompt))

    result = getResultFromChatGPT(prompt)
    global gpt_key_index
    gpt_key_index = 1 if gpt_key_index >= 3 else gpt_key_index + 1

    print("result: {}\n".format(result))
    try:
        lang, sql_query, elastic_query = extract_lang_sql_elastic_from_result(
            result)
        rows, cols = exec_elasticsearch_search(elastic_query)
        # Tries count = 3
        try_count = 3
        if len(rows) == 0:
            while try_count > 0:
                if len(rows) == 0:
                    # sleep for 3 second
                    time.sleep(3)
                    openai.api_key = get_gpt_key(gpt_key_index)
                    result = getResultFromChatGPT(prompt)
                    gpt_key_index = 1 if gpt_key_index >= 3 else gpt_key_index + 1
                    lang, sql_query, elastic_query = extract_lang_sql_elastic_from_result(
                        result)
                    rows, cols = exec_elasticsearch_search(elastic_query)
                try_count -= 1
    except Exception as e:
        # Handle the exception
        print(f"An error occurred(gpt_solution_qa): {str(e)}")
        rows = []
        cols = []
        # Retry get query from ChatGPT
        result = getResultFromChatGPT(prompt)
        gpt_key_index = 1 if gpt_key_index >= 3 else gpt_key_index + 1
        lang, sql_query, elastic_query = extract_lang_sql_elastic_from_result(
            result)
        rows, cols = exec_elasticsearch_search(elastic_query)

    if (lang == "English"):
        return rows, cols, lang, sql_query, elastic_query
    else:
        lang_prompt = """Translate the following Objects to {} language:
        ''
        cols: {}
        ''
        rows: {}

        Output the results in the following format:
        1. cols: 
        2. rows:
        """.format(lang, json.dumps(cols), json.dumps(rows[0]))

        time.sleep(3)
        print("lang_prompt: {}\n".format(lang_prompt))
        translated_result = getResultFromChatGPT(lang_prompt)
        gpt_key_index = 1 if gpt_key_index >= 3 else gpt_key_index + 1
        print("translated_result===> : {}\n".format(translated_result))
        translated_rows, translated_cols = extract_cols_rows_from_translated_result(
            translated_result)

        return [translated_rows], translated_cols, lang, sql_query, elastic_query


def extract_cols_rows_from_translated_result(translated_result):
    # Replace the Chinese comma with a regular comma
    translated_result = translated_result.replace("，", ",")

    # Extracting cols using regular expression
    cols_match = re.search(r'cols: (.+)', translated_result)
    if cols_match:
        cols = cols_match.group(1).strip()

    # Extracting rows using regular expression
    rows_match = re.search(r'rows: (.+)', translated_result)
    if rows_match:
        rows = rows_match.group(1).strip()

    print("cols =", cols)
    print("rows =", rows)

    remove_characters = ['[', ']', '"']

    # Create a translation table to remove the specified characters
    translation_table = str.maketrans('', '', ''.join(remove_characters))

    # Remove the characters using the translation table
    cols = cols.translate(translation_table)
    rows = rows.translate(translation_table)

    cols = [col.strip() for col in cols.split(',')]
    rows = [row.strip() for row in rows.split(',')]

    print("cols =", cols)
    print("rows =", rows)

    return rows, cols


def getResultFromChatGPT(prompt):
    request = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.2,
        max_tokens=3500
    )
    result = request.choices[0].text
    print(request)

    # # Remove any characters before the "SELECT" word in the SQL query
    # select_index = query.find("SELECT")
    # if select_index != -1:
    #     query = query[select_index:]

    return result


def convert_to_table(rows, cols):
    # Start building the HTML table
    html_table = "<table>\n"

    # Add the table header
    html_table += "  <tr>\n"
    for column in cols:
        html_table += f"    <th>{column}</th>\n"
    html_table += "  </tr>\n"

    # Loop through the rows
    for row in rows:
        html_table += "  <tr>\n"
        for value in row:
            html_table += f"    <td>{value}</td>\n"
        html_table += "  </tr>\n"

    # Close the HTML table
    html_table += "</table>"

    return html_table


def extract_lang_sql_elastic_from_result(result):
    # Extracting values using regular expressions
    sql_query_match = re.search(r'SQL Server query: (.+)', result)
    elastic_query_match = re.search(
        r'Elastic Search query: (\{.*?\})(?=\s*3\.)', result, re.DOTALL)
    language_match = re.search(r'Detected language: (.+)', result)

    if sql_query_match and elastic_query_match and language_match:
        sql_query = sql_query_match.group(1)
        elastic_query = elastic_query_match.group(1)
        language = language_match.group(1)

        print("sql_query =", sql_query)
        print("elastic_query =", elastic_query)
        print("language =", language)
    else:
        print("Data extraction failed.")

    return language, sql_query, elastic_query

# prmt =  """请找一个大于2欧姆的电阻"""
# apply_gpt_sql_elastic_translate_query(prmt, 'VIEW_DATA_PART')
