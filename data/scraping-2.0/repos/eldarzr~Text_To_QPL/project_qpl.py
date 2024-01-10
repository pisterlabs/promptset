#!/usr/bin/env python
# coding: utf-8

import sqlite3
import os
import json
import sys
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import openai
import requests
import time
import re


# read dataset - schemas, samples (tuples of the form schema_id, query, expected sql)
# best is to read all of it into a well defined Python data structure
def read_train_json(file_name):
    f = open(f'{file_name}.json')
    # list of schemas
    schemas = json.load(f)
    # Closing file
    f.close()
    return schemas


# execute query on schema and collect result set (best is to collect it into a pandas dataframe)
def collect_to_pd(schemas):
    # Create a connection to the SQLite database
    # If the database does not exist, it will be created
    data_set = []
    for schema in schemas:
        db_name = schema['db_id']
        query = schema['query']
        question = schema['question']
        connection = sqlite3.connect(f'database//{db_name}//{db_name}.sqlite')
        connection.text_factory = lambda b: str(b, 'utf-8', 'ignore')

        # Use pandas to run the SQL query and store the results in a DataFrame
        df = pd.read_sql_query(query, connection)

        connection.close()

        data_set.append({'db_id': db_name, 'question': question, 'query': query, 'exec': df, 'gpt_results': []})
    return data_set


# # Global Variables

args = sys.argv
if len(args) > 1:
    spider_database_folder_path = args[1]
else:
    spider_database_folder_path = r"C:\Users\אלדר זריהן\Documents\לימודים\סמסטר ו'\חזית\text to sql proj\spider"
if len(args) > 2:
    api_key = args[2]
else:
    api_key = None
os.chdir(spider_database_folder_path)
seed = 3248524
train_spider = read_train_json("train_spider")
train_others = read_train_json("train_others")
dev = read_train_json("dev")
dev_examples = read_train_json("dev_examples")
train = read_train_json("train")
tables = {schema["db_id"]: schema for schema in read_train_json("tables")}


# # Verbose

def get_column_values_from_question(table_name, db_name, column, question):
    connection = sqlite3.connect(f'database\\{db_name}\\{db_name}.sqlite')
    connection.text_factory = lambda b: str(b, 'utf-8', 'ignore')
    alpha_numeric_column = f'"{column}"'
    df = pd.read_sql_query(f"SELECT {alpha_numeric_column} from {table_name}", connection)
    appears1 = set(map(lambda y: y.lower(), filter(lambda x: x.lower() in question.lower().split(" "), [str(x) for x in
                                                                                                        df[column]])))

    appears2 = set(filter(lambda x: x.lower() in appears1, question.lower().split(" ")))

    connection.close()
    appears = appears1.intersection(appears2)
    return appears


def get_tables_info(schema, with_types=True):
    output_string = ""
    for i, table_name in enumerate(schema["table_names_original"]):
        if not with_types:
            column_names = [schema["column_names_original"][j][1] for j in range(len(schema["column_names_original"]))
                            if schema["column_names_original"][j][0] == i]
            columns_names_string = ", ".join(column_names)
            table_string = f"\nTable {i + 1} is {table_name}, and its column names are: {columns_names_string}. "
            output_string += table_string
        else:
            column_to_type = {schema["column_names_original"][j][1]: schema["column_types"][j] for j in
                              range(len(schema["column_names_original"])) if schema["column_names_original"][j][0] == i}
            column_type_string = ", ".join(
                [f"{column} (Type is {column_to_type[column]})" for column in column_to_type])
            table_string = f"\nTable {i + 1} is {table_name}, and its column names and types are: {column_type_string}. "
            output_string += table_string
    return output_string


def build_verbose_prompt(query_dict, with_values=True, with_types=True):
    schema = tables[query_dict["db_id"]]
    num_of_tables = len(schema["table_names"])
    titles = ", ".join(schema["table_names_original"])
    intro = """This is a task converting text into SQL statement.
We will first give the dataset schema and then ask a question in text.
You are asked to generate an SQL statement.
Here is the test question to be answered: Let us take a question and turn it into a SQL statement about database tables. """
    schema_spesific_info = "There are {number_of_tables} tables. Their titles are: {titles}. ".format(
        number_of_tables=num_of_tables, titles=titles)
    tables_info = get_tables_info(schema, with_types)
    primary_keys = ", ".join([
        f"{schema['column_names_original'][i][1]} from Table {schema['table_names_original'][schema['column_names'][i][0]]}"
        for i in schema['primary_keys']])
    primary_keys_info = f"\nThe primary keys are: {primary_keys}."
    foreign_keys = ", ".join([
        f"{schema['column_names_original'][i][1]} from Table {schema['table_names_original'][schema['column_names'][i][0]]} is equivalent to {schema['column_names_original'][j][1]} from Table {schema['table_names_original'][schema['column_names'][j][0]]}"
        for i, j in schema['foreign_keys']])
    foreign_keys_info = f"\nThe foreign keys are: {foreign_keys}. Use foreign keys to join Tables. "
    if with_values:
        relevant_values_list = []
        for i in range(1, len(schema['column_names'])):
            column_ralevent_values = get_column_values_from_question(
                schema['table_names_original'][schema['column_names'][i][0]], query_dict['db_id'],
                schema['column_names_original'][i][1], query_dict['question'])
            if column_ralevent_values:
                relevant_values_list.append(
                    f"Table {schema['table_names_original'][schema['column_names'][i][0]]} Column {schema['column_names_original'][i][1]} have values: " + ", ".join(
                        column_ralevent_values))
        relevant_values_string = "; ".join(relevant_values_list)
        relevant_values_string = f"Columns with relevant values: {relevant_values_string}; Only use columns with relevant values to generate SQL. "
    question_info = f"\nLet us take a text question and turn it into a SQL statement about database tables. The question is: {query_dict['question']} The corresponding SQL is:"
    final_string = f"{intro}{schema_spesific_info}{tables_info}{primary_keys_info}{foreign_keys_info}"
    if with_values:
        final_string += relevant_values_string
    final_string += question_info
    return final_string


# # Concise

def concise_values(db_id, tables, columns, question, with_values=True):
    _prompt = ''
    for i, table in enumerate(tables):
        _prompt += f" | {table} : "
        cols_with_vals = []
        cols = list(map(lambda pair: pair[1], filter(lambda pair: pair[0] == i, columns)))
        # missing values
        for col in cols:
            col_with_val = col
            if with_values:
                appears = get_column_values_from_question(tables[i], db_id, col, question)
                if appears:
                    col_with_val += ' ( ' + ' , '.join(appears) + ' )'
            cols_with_vals.append(col_with_val)
        if len(cols_with_vals) > 0:
            _prompt += " , ".join(cols_with_vals)
    _prompt += '\n'
    return _prompt


def concise_types(tables, columns, types):
    _prompt = '[Column names(type)]: \n'
    for i, table in enumerate(tables):
        _prompt += f"{table} : "
        cols = []
        for j, col_pair in enumerate(columns):
            if col_pair[0] == i:
                cols.append(f"{col_pair[1]} ({types[j]})")
        _prompt += " | ".join(cols)
        if i != len(tables) - 1:
            _prompt += f"\n"
    _prompt += '\n'
    return _prompt


def concise_primary_key(tables, columns, primary_keys):
    _prompt = '[Primary Keys]: \n'
    for i, table in enumerate(tables):
        _prompt += f"{table} : "
        cols = []
        for key in primary_keys:
            if columns[key][0] == i:
                cols.append(columns[key][1])
        _prompt += " , ".join(cols)
        if i != len(tables) - 1:
            _prompt += f"\n"
    _prompt += '\n'
    return _prompt


def concise_foreign_key(tables, columns, foreign_keys):
    _prompt = '[Foreign Keys]: \n'
    cols = []
    for key in foreign_keys:
        col_1 = columns[key[0]]
        col_2 = columns[key[1]]
        table_1 = tables[col_1[0]]
        table_2 = tables[col_2[0]]
        cols.append(f"{table_1} : {col_1[1]} equals {table_2} : {col_2[1]}")
        # if i != len(tables)-1:
        #     _prompt += f" | "
    _prompt += "\n".join(cols)
    _prompt += '\n'
    return _prompt


def build_concise(db, question, with_values, with_types):
    prompt = "This is a task converting text into SQL statement. "              "We will first given the dataset " \
             "schema and then ask "              "a question in text. You are asked to generate SQL " \
             "statement. Here is the test question to be anwered: "              "Convert text to SQL: \n" \
             "[Schema (values)]: | "
    db_id = db['db_id']
    tables = db['table_names_original']
    columns = db['column_names_original']
    types = db['column_types']
    primary_keys = db['primary_keys']
    foreign_keys = db['foreign_keys']

    prompt += f"{db_id}"

    # values
    prompt += concise_values(db_id, tables, columns, question, with_values)

    # types
    if with_types:
        prompt += concise_types(tables, columns, types)

    # primary keys
    prompt += concise_primary_key(tables, columns, primary_keys)

    # foreign keys
    prompt += concise_foreign_key(tables, columns, foreign_keys)

    prompt += f"[Q]: {question} \n [SQL]: "

    return prompt.lower()


def validate_qpl(schema, qpl, url="http://localhost"):
    json_data = {
        'qpl': f'{schema} | {qpl}',
    }
    response = requests.post(f'{url}:8000/validate', json=json_data)
    return response.status_code == 200 and response.text == 'true'


def submit_qpl(schema, qpl, url="http://localhost"):
    json_data = list(qpl.split('\n'))
    response = requests.post(f'{url}:8000/{schema}/qpl', json=json_data)
    if response.status_code == 200:
        data = json.loads(response.text)

        try:
            # Convert the 'result' key to a DataFrame
            df = pd.DataFrame(data['result'])
            df = df.sort_values(by=list(df.columns))
            df = df[sorted(df.columns)]

            # Now, df contains the desired data
            for col in df.columns:
                # Check if the column is of type 'object' (which usually means it's a string column in pandas)
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip()
            return df
        except:
            pass
    return None


def qpl_general_info():
    return """QPL is a formalism used to describe data retrieval operations over an SQL schema in a modular manner.
A QPL plan is a sequence of instructions for querying tabular data to answer a natural language question.
Forget everything you know about SQL, only use the following explanations.

A schema is specified as a list of <table> specification in the format:
<table>: <comma separated list of columns>

A plan contains a sequence of operations.
All operations return a stream of tuples.
All operations take as input either a physical table from the schema (for the Scan operation) or the output of other operations.

Your task is to learn the QPL BNF and the examples and to provide only the QPL plan according to the schema and the 
question I will give you below.

This is the formal specification for each operation:

"""


def qpl_bnf():
    return """<qpl> ::= <line>+
<line> ::= # <integer> = <operator>
<operator> ::= <scan> | <aggregate> | <filter> | <sort> | <join> | <except> | <intersect> | <union>
<scan> ::= Scan Table [ <table-name> ] <predicate>? <distinct>? <output-list-non-qualified>
<aggregate> ::= Aggregate [ <input> ] <group-by>? <output-list-non-qualified>
<filter> ::= Filter [ <input> ] <predicate> <distinct>? <output-list-non-qualified>
<sort> ::= Sort [ <input> ] <order-by> <output-list-non-qualified>
<join> ::= Join [ <input> , <input> ] <predicate>? <distinct>? <output-list-qualified>
<except> ::= Except [ <input> , <input> ] <predicate> <output-list-qualified>
<intersect> ::= Intersect [ <input> , <input> ] <predicate>? <output-list-qualified>
<union> ::= Union [ <input> , <input> ] <output-list-qualified>
<group-by> ::= GroupBy [ <column-name> (, <column-name>)* ]
<order-by> ::= OrderBy [ <column-name> <direction> (, <column-name> <direction>)* ]
<direction> ::= ASC | DESC
<predicate> ::= Predicate [ <comparison> (AND | OR <comparison)* ]
<distinct> ::= Distinct [ true ]
<output-list-non-qualified> ::= Output [ <column-name> (, <column-name>)* ]
<output-list-qualified> ::= Output [ <qualified-column-name> (, <qualified-column-name>)* ]
<qualified-column-name> ::= # <number> . <column-name>

"""


def qpl_schema_info(db_id, tables, columns, question, with_values):
    schema = 'Schema:'
    concise_vals = concise_values(db_id, tables, columns, question, with_values)
    qpl_vals = concise_vals.replace('|', '\ntable').replace(';', '')
    return schema + qpl_vals


def qpl_types(tables, columns, types):
    con_types = concise_types(tables, columns, types)
    return '\n'.join(["Types of columns:"] + con_types.replace('|', ',').replace(';', '').split('\n')[1:])


def qpl_primary_keys(tables, columns, primary_keys):
    concise_primary = concise_primary_key(tables, columns, primary_keys)
    return '\n'.join(["Primary keys of columns:"] +
                     concise_primary.replace('|', ',').replace(';', '').split('\n')[1:])


def qpl_foreign_keys(tables, columns, foreign_keys):
    concise_foreign = concise_foreign_key(tables, columns, foreign_keys)
    return '\n'.join(["Foreign keys of columns:"] +
                     concise_foreign.replace('|', ',').replace(';', '').split('\n')[1:])


def get_diff_examples(diff,num_examples):
    diff_schemas = [s for s in train if s["difficulty"] == diff]
    random.seed(seed)
    random_indices = random.sample(range(len(diff_schemas)), num_examples)
    return [diff_schemas[i] for i in random_indices]


def get_example_schemas():
    examples = []
    diff = ["hard", "extra"]
    for d in diff:
        num_examples_for_diff = 3
        examples += get_diff_examples(d, num_examples_for_diff)
    return examples


def build_qpl_examples(with_values, with_types):
    examples_schemas = get_example_schemas()
    output_string = ""
    for i, qpl_example in enumerate(examples_schemas):
        db_id = qpl_example['db_id']
        db = tables[db_id]
        table_names = db['table_names_original']
        columns = db['column_names_original']
        types = db['column_types']
        primary_keys = db['primary_keys']
        foreign_keys = db['foreign_keys']

        example_prompt = f"Example {i + 1}:\n\n"
        question = qpl_example['question']
        example_prompt += qpl_schema_info(db_id, table_names, columns, question, with_values)
        if with_types:
            example_prompt += qpl_types(table_names, columns, types)
        example_prompt += qpl_primary_keys(table_names, columns, primary_keys)
        example_prompt += qpl_foreign_keys(table_names, columns, foreign_keys)
        example_prompt += f"Question:\n{qpl_example['question']}\n\n"
        example_prompt += f"QPL Plan:\n{qpl_example['qpl']}\n\n"
        output_string += example_prompt
    return output_string


def build_qpl_prompt(db, question, with_values, with_types):
    tables = db['table_names_original']
    columns = db['column_names_original']
    db_id = db['db_id']
    types = db['column_types']
    primary_keys = db['primary_keys']
    foreign_keys = db['foreign_keys']

    general_info = qpl_general_info()

    bnf = qpl_bnf()

    prompt = general_info + bnf

    # here goes 6 examples
    prompt += build_qpl_examples(with_values, with_types)

    prompt += "Now your turn:\n\n"
    prompt += qpl_schema_info(db_id, tables, columns, question, with_values)
    if with_types:
        prompt += qpl_types(tables, columns, types)
    prompt += qpl_primary_keys(tables, columns, primary_keys)
    prompt += qpl_foreign_keys(tables, columns, foreign_keys)
    prompt += f"Question:\n{question}\n\n"
    prompt += "the following QPL Plan is:"
    return prompt


def build_prompt(query_dictionary, design, with_values=False, with_types=False):
    prompts_file_name = 'values_with_types.json' if with_types else 'values_without_types.json'
    prompt = None
    if design == "concise":
        prompt = build_concise(tables[query_dictionary['db_id']], query_dictionary['question'], with_values, with_types)
        prompts_file_name = "concise_" + prompts_file_name
    elif design == 'verbose':
        prompt = build_verbose_prompt(query_dictionary, with_values, with_types)
        prompts_file_name = "verbose_" + prompts_file_name
    elif design == 'qpl':
        prompt = build_qpl_prompt(tables[query_dictionary['db_id']], query_dictionary['question'], with_values,
                                  with_types)
        prompts_file_name = "qpl_" + prompts_file_name
    if prompt is not None:
        with open(prompts_file_name, 'a') as json_file:
            json.dump(prompt, json_file)
    return prompt


# # GPT Mock

def gpt_mock(schema, prompt=None):
    table = schema['table_names_original'][0]
    column = schema['column_names_original'][1][1]
    return [f"#1 = Scan Table [ {table} ] Output [ {column} ] ; #2 = Scan Table [ {table} ] Output [ {column} ]"]


def gpt_real(prompt, temp=0.4, n=3):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        print("attempt to query gpt")
        openai.api_key = api_key
        model = "gpt-3.5-turbo-16k"
        response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=450, temperature=temp, n=n)
        print(response['choices'][0]['message']['content'])
        return list(map(lambda q: q['message']['content'], response['choices']))
    except openai.error.RateLimitError as e:
        print("exception in gpt function, waiting...")
        msg = e.user_message
        print(msg)
        match = re.search(r'\bplease try again in (\d+)s\b', msg, re.IGNORECASE)
        number = int(match.group(1)) if match else 25
        time.sleep(number)
        return gpt_real(prompt)


# takes n and return n queries dictionaries
def get_random_queries(queries_dicts, seed, num_extra_hard_schemas, num_hard_schemas, mode):
    schemas_with_difficulty = read_train_json("dev_examples")
    extra_hard_schemas = [s for s in schemas_with_difficulty if s["difficulty"] == "extra"]
    hard_schemas = [s for s in schemas_with_difficulty if s["difficulty"] == "hard"]
    random.seed(seed)
    random_indices_extra_hard = random.sample(range(len(extra_hard_schemas)), num_extra_hard_schemas)
    random_indices_hard = random.sample(range(len(hard_schemas)), num_hard_schemas)
    extra_hard_schemas_random = [extra_hard_schemas[i] for i in random_indices_extra_hard]
    if [s for s in extra_hard_schemas_random if s["db_id"] == "concert_singer"]:
        raise ValueError("concert_singer too big for this code")
    hard_schemas_random = [hard_schemas[i] for i in random_indices_hard]
    if [s for s in hard_schemas_random if s["db_id"] == "concert_singer"]:
        raise ValueError("concert_singer too big for this code")
    queries_dicts_extra_hard = [qd for qd in queries_dicts for qd_wd in extra_hard_schemas_random if
                                qd["question"] == qd_wd["question"]]
    for query_dict in queries_dicts_extra_hard:
        db_name = query_dict['db_id']
        question = query_dict['question']
        qpl_dev = [qpl for qpl in dev_examples if qpl['db_id'] == db_name and qpl['question'] == question][0]
        query_dict['qpl'] = qpl_dev['qpl']
        if mode == 'qpl':
            query_dict["exec"] = submit_qpl(db_name, qpl_dev['qpl'])
    queries_dicts_hard = [qd for qd in queries_dicts for qd_wd in hard_schemas_random if
                          qd["question"] == qd_wd["question"]]
    for query_dict in queries_dicts_hard:
        db_name = query_dict['db_id']
        question = query_dict['question']
        qpl_dev = [qpl for qpl in dev_examples if qpl['db_id'] == db_name and qpl['question'] == question][0]
        query_dict['qpl'] = qpl_dev['qpl']
        if mode == 'qpl':
            query_dict["exec"] = submit_qpl(db_name, qpl_dev['qpl'])
    return queries_dicts_extra_hard + queries_dicts_hard


# return dataframe of execution of sql query
def exec_sql(db_name, query):
    connection = sqlite3.connect(f'database\\{db_name}\\{db_name}.sqlite')
    connection.text_factory = lambda b: str(b, 'utf-8', 'ignore')
    try:
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df
    except:
        return None


# return histogram of sql execution results
def create_histogram(dataframes):
    histogram = []

    for df in dataframes:
        if df is None or df.empty:
            continue
        filtered = list(filter(lambda h: compare_sql_exec(h['df'], df), histogram))
        if len(filtered):
            for h in filtered:
                h['value'] = h.get('value', 0) + 1
        else:
            histogram.append({'df': df, 'value': 1})

    return histogram


# return most common sql query
def most_common_execution(dataframes: list):
    histogram = create_histogram(dataframes)
    sorted_hist = list(sorted(histogram, key=lambda h: h['value'], reverse=True))
    return sorted_hist[0]['df'] if len(sorted_hist) > 0 else None


# return if 2 sql execution are equals
def compare_sql_exec(df1, df2, is_ordered=False):

    if df1 is not None and df1.empty:
        df1 = None

    if df2 is not None and df2.empty:
        df2 = None

    if df1 is None or df2 is None:
        if df1 is None and df2 is None:
            return True
        return False

    # there is 'ORDER BY' in the query

    if is_ordered:
        values_df1 = [df1.iloc[i].values for i in range(df1.shape[0])]
        values_df2 = [df2.iloc[i].values for i in range(df2.shape[0])]
    else:
        values_df1 = sorted([df1.iloc[i].values for i in range(df1.shape[0])], key=lambda x: x[0])
        values_df2 = sorted([df2.iloc[i].values for i in range(df2.shape[0])], key=lambda x: x[0])

    # not the same len of columns
    if len(values_df1) != len(values_df2):
        return False

    return all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(values_df1, values_df2))


# # Json Functions

def save_to_file_json(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file)


def load_from_file_json(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the file doesn't exist


# json cannot same dataframes so we save the query dictionay without the execution reusults
def add_exec_result_after_load_sql(query_dict):
    db_name = query_dict["db_id"]
    question = query_dict["question"]
    connection = sqlite3.connect(f'database//{db_name}//{db_name}.sqlite')
    connection.text_factory = lambda b: str(b, 'utf-8', 'ignore')
    df = pd.read_sql_query(query_dict["query"], connection)
    query_dict["exec"] = df

    gpt_results = query_dict["gpt_results"]
    for gpt_result in gpt_results:
        gpt_result["query_result"] = pd.read_sql_query(gpt_result["query"], connection)
    connection.close()


def add_exec_result_after_load_qpl(query_dict):
    db_name = query_dict["db_id"]
    question = query_dict["question"]
    qpl_dev = [qpl for qpl in dev_examples if qpl['db_id'] == db_name and qpl['question'] == question][0]
    qpl_dev['qpl'] = qpl_dev['qpl'].replace(' ; ', '\n')
    qpl_dev['qpl'] = '\n'.join([line.strip() for line in qpl_dev['qpl'].splitlines()])
    query_dict["exec"] = submit_qpl(db_name, qpl_dev['qpl'])

    gpt_results = query_dict["gpt_results"]
    for gpt_result in gpt_results:
        gpt_result["query"] = gpt_result["query"].replace(' ; ', '\n').replace('Average', 'Avg')
        gpt_result['query'] = '\n'.join([line.strip() for line in gpt_result['query'].splitlines()])
        gpt_result["query_result"] = submit_qpl(db_name, gpt_result["query"])


def add_exec_result_after_load(query_dict, mode):
    if mode == 'qpl':
        return add_exec_result_after_load_qpl(query_dict)
    return add_exec_result_after_load_sql(query_dict)


def clean_query_exec_before_saving(query_dict):
    query_dict_copy = {}
    for key in query_dict:
        if key == "exec" or key == "gpt_results":
            continue
        query_dict_copy[key] = query_dict[key]
    query_dict_copy["gpt_results"] = []
    gpt_results = query_dict["gpt_results"]
    gpt_res_copy = {}
    for gpt_res in gpt_results:
        for key in gpt_res:
            if key == "query_result":
                continue
            gpt_res_copy[key] = gpt_res[key]
        query_dict_copy["gpt_results"].append(gpt_res_copy)
        gpt_res_copy = {}
    return query_dict_copy


def check_if_already_exists(mode, query_dict, with_values, with_types):
    if "gpt_results" in query_dict:
        gpt_res = queries_dict["gpt_result"]
        for gr in gpt_res:
            if (mode == gpt_res["mode"]) and (gr["with_values"] == with_values) and (gr["with_types"] == with_types):
                return True
    return False


def add_gpt_results_to_query_dict(mode, query_dict, with_values, with_types, difficulty=None):
    if mode == "concise" or mode == "MixPrompt":
        concise_prompt = build_prompt(query_dict, "concise", with_values, with_types)
        concise_gpt_sql_queries = gpt_real(prompt=concise_prompt)  # gpt(prompt = cocise_prompt)
        for gpt_query in concise_gpt_sql_queries:
            sql_res = exec_sql(query_dict["db_id"], gpt_query)
            if sql_res is None:
                continue
            gpt_result = {"type": "concise", "query": gpt_query, "query_result": sql_res, "with_values": with_values,
                          "with_types": with_types}
            query_dict["gpt_results"].append(gpt_result)
    if mode == "verbose" or mode == "MixPrompt":
        verbose_prompt = build_prompt(query_dict, "verbose", with_values, with_types)
        verbose_gpt_sql_queries = gpt_real(prompt=verbose_prompt)  # gpt_real(verbose = verbose_prompt)
        for gpt_query in verbose_gpt_sql_queries:
            sql_res = exec_sql(query_dict["db_id"], gpt_query)
            if sql_res is None:
                continue
            gpt_result = {"type": "verbose", "query": gpt_query, "query_result": sql_res, "with_values": with_values,
                          "with_types": with_types}
            query_dict["gpt_results"].append(gpt_result)
    if mode == "qpl":
        qpl_prompt = build_prompt(query_dict, "qpl", with_values, with_types)
        query_dict['difficulty'] = difficulty
        # qpl_gpt_sql_queries = gpt_mock(tables[query_dict['db_id']], prompt = qpl_prompt)  # gpt_real(verbose = verbose_prompt)
        qpl_gpt_sql_queries = gpt_real(prompt=qpl_prompt)  # gpt_real(verbose = verbose_prompt)
        qpl_gpt_sql_queries = [qpl[0:-1 if qpl[-1] == ';' else len(qpl)].strip().replace('\n', ' ').replace('  ',
                                                                                                            ' ') for
                               qpl in
                               qpl_gpt_sql_queries]
        for gpt_query in qpl_gpt_sql_queries:
            is_qpl_valid = validate_qpl(query_dict["db_id"], gpt_query)
            qpl_res = submit_qpl(query_dict["db_id"], gpt_query) if is_qpl_valid else None
            qpl_len = gpt_query.count(';')+1
            gpt_result = {"type": "qpl", "query": gpt_query, "query_result": qpl_res, "with_values": with_values,
                          "with_types": with_types, "is_valid": is_qpl_valid, "qpl_len": qpl_len}
            query_dict["gpt_results"].append(gpt_result)
    return query_dict


def main(mode="qpl", dataset=dev):
    queries_dicts = collect_to_pd(dataset)
    n = 25
    queries_dicts_random = get_random_queries(queries_dicts, seed, num_extra_hard_schemas=20, num_hard_schemas=5,
                                              mode=mode)
    prompt_additional_params = [(False, False,), (False, True,), (True, False,), (True, True,)]
    same_h = 0
    same_e = 0
    if mode == 'qpl':
        file_name = "question_to_query_dictionaries_qpl.json"
    else:
        file_name = "question_to_query_dictionaries.json"
    loaded_data = load_from_file_json(file_name)
    total_until_now = 0
    for i, query_dict in tqdm(enumerate(queries_dicts_random), total=len(queries_dicts_random)):
        key = f"{mode} - {query_dict['question']}"
        start = 0
        if key in loaded_data:
            query_dict = loaded_data[key]
            add_exec_result_after_load(query_dict, mode)
        else:
            for idx in range(start, len(prompt_additional_params)):
                with_values = prompt_additional_params[idx][0]
                with_types = prompt_additional_params[idx][1]
                difficulty = 'hard' if i >= 20 else 'extra'
                query_dict = add_gpt_results_to_query_dict(mode, query_dict, with_values, with_types, difficulty)
        execution_dataframes = [result['query_result'] for result in query_dict['gpt_results']]
        max_result = most_common_execution(execution_dataframes)
        # sort maxResult based on sort by in query
        is_ordered = "order by" in query_dict["query"].lower()
        compare_results_res = compare_sql_exec(max_result, query_dict["exec"], is_ordered)

        if mode == 'qpl':
            query_col_names = list(query_dict['exec'].columns) if query_dict['exec'] is not None else []
            query_len = query_dict["qpl"].count(';') + 1
            query_dict["query_len"] = query_len
            query_dict["col_names"] = ','.join(query_col_names)
            for gpt_res in query_dict['gpt_results']:
                compare_common = compare_sql_exec(max_result, gpt_res['query_result'], is_ordered)
                compare_correct = compare_sql_exec(query_dict["exec"], gpt_res['query_result'], is_ordered)
                res_col_names = list(gpt_res['query_result'].columns) if gpt_res['query_result'] is not None else []
                inter = set(res_col_names).intersection(set(query_col_names))
                gpt_res["is_common"] = compare_common
                gpt_res["is_correct"] = compare_correct
                gpt_res["col_names"] = ','.join(res_col_names)
                gpt_res["intersection"] = ','.join(inter)

        if key not in loaded_data:
            query_dict_without_exec = clean_query_exec_before_saving(query_dict)
            loaded_data[key] = query_dict_without_exec
            for key in loaded_data:
                loaded_data[key] = clean_query_exec_before_saving(loaded_data[key])
            save_to_file_json(file_name, loaded_data)

        total_until_now += 1
        if compare_results_res:
            if i >= 20:
                same_h += 1
            else:
                same_e += 1
            if max_result is None:
                print(f"{((i+1)*12) - 10} {query_dict['question']}")
    rate_h = float(same_h) / float(5)
    rate_e = float(same_e) / float(20)
    print(f"mode -{mode}, rate hard = {rate_h}")
    print(f"mode -{mode}, rate extra = {rate_e}")


if __name__ == '__main__':
    if len(args) > 3:
        main(args[3])
    else:
        main()
