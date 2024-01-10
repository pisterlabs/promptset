import locale
import copy
from collections import OrderedDict
import json
import os
import re
import time
from typing import Any, List

import pandas as pd
import openai
from openai import OpenAIError
from tqdm import tqdm
import ast
import sys

from mitaskem.src.util import *
from mitaskem.src.gpt_interaction import *
from mitaskem.src.mira_dkg_interface import *

GPT_KEY = os.environ.get('OPENAI_API_KEY')

# from automates.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet
# from automates.gromet.query import query

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def index_text_path(text_path: str) -> str:
    fw = open(text_path + "_idx", "w")
    with open(text_path) as fp:
        for i, line in enumerate(fp):
            fw.write('%d\t%s' % (i, line))
    fw.close()
    return text_path + "_idx"

def index_text(text: str) -> str:
    idx_text = ""
    tlist = text.splitlines()
    # print(tlist)
    for i, line in enumerate(tlist):
        if i==len(tlist)-1 and line== "":
            break
        idx_text = idx_text + ('%d\t%s\n' % (i, line))
    return idx_text

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import LatexTextSplitter
from langchain.schema.messages import HumanMessage, SystemMessage

def get_gpt_match(prompt, key, model="gpt-3.5-turbo"):
    # model="text-davinci-003" does not use completion
    llm = ChatOpenAI(model_name=model, openai_api_key=key, temperature=0, max_tokens=256)
    completion = llm.invoke(input=[HumanMessage(content=prompt)])
    return completion.content.strip()

def get_gpt4_match(prompt, key, model="gpt-4"):    
    llm = ChatOpenAI(model_name=model, openai_api_key=key, temperature=0)
    completion = llm.invoke(input=[HumanMessage(content=prompt)])
    return completion.content

def read_text_from_file(text_path):
    text_file = open(text_path, "r")
    prompt = text_file.read()
    return prompt


# Get gpt-3 prompt with arizona-extraction, ontology terms and match targets
def get_prompt(vars, terms, target):
    text_file = open("prompts/prompt.txt", "r")
    prompt = text_file.read()
    text_file.close()

    vstr = ''
    vlen = len(vars)
    i = 1;
    for v in vars:
        vstr += str(i) + " (" + str(v[1]) + ", " + str(v[2]) + ")\n"
        i += 1;
    # print(vstr)
    tstr = '[' + ', '.join(terms) + ']'
    tlen = len(terms)
    # print(tstr)
    prompt = prompt.replace("[VAR]", vstr)
    prompt = prompt.replace("[VLEN]", str(vlen))
    prompt = prompt.replace("[TERMS]", tstr)
    prompt = prompt.replace("[TLEN]", str(tlen))
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt


# Get gpt-3 prompt with formula, code terms and match formula targets
def get_code_formula_prompt(code, formula, target):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/code_formula_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[CODE]", code)
    prompt = prompt.replace("[FORMULA]", formula)
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt


# Get gpt-3 prompt with variable and datasets, and match variable target with the best data columns
def get_var_dataset_prompt(vars, dataset, target):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/var_dataset_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[DESC]", vars)
    prompt = prompt.replace("[DATASET]", dataset)
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt

def get_var_dataset_prompt_simplified(vars, dataset):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/var_dataset_prompt_simplified.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[DESC]", vars)
    prompt = prompt.replace("[DATASET]", dataset)
    # print(prompt)
    return prompt


def get_rank_dkg_prompt(target, set):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/rank_dkg_terms.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TARGET]", target)
    prompt = prompt.replace("[SET]", set)
    # print(prompt)
    return prompt
#
def get_mit_arizona_var_prompt(mit, arizona):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/mit_arizona_var_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[MIT]", mit)
    prompt = prompt.replace("[ARIZONA]", arizona)
    # print(prompt)
    return prompt

# Get gpt-3 prompt with formula, and match variable targets
def get_var_formula_prompt(desc, var):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/var_formula_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[DESC]", desc)
    prompt = prompt.replace("[TARGET]", var)
    # print(prompt)
    return prompt

# Get gpt-3 prompt with formula, and match variable targets
def get_formula_var_prompt(formula):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/formula_var_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[FORMULA]", formula)
    # print(prompt)
    return prompt

def get_all_desc_formula_prompt(all_dsec, latex_var):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/all_desc_formula_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[DESC]", all_dsec)
    prompt = prompt.replace("[TARGET]", latex_var)
    return prompt

# Get gpt-3 prompt with formula, code terms and match formula targets
def get_code_text_prompt(code, text, target):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/code_text_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[CODE]", code)
    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt

def get_text_param_prompt(text):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/text_param_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    return prompt

def get_text_var_prompt(text):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/text_var_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    return prompt


# Get gpt-3 prompt with code, dataset and match function targets
def get_code_dataset_prompt(code, dataset, target):
    text_file = open(os.path.join(os.path.dirname(__file__), "prompts/code_dataset_prompt.txt"), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[CODE]", code)
    prompt = prompt.replace("[DATASET]", dataset)
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt


def get_csv_doc_prompt(schema, stats, doc, dataset_name, doc_name):
    text_file = open(os.path.join(os.path.dirname(__file__), "prompts/dataset_profiling_prompt.txt"), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[SCHEMA]", schema)
    prompt = prompt.replace("[STATS]", json.dumps(stats))
    prompt = prompt.replace("[DOC]", doc)
    prompt = prompt.replace("[DATASET_NAME]", dataset_name)
    prompt = prompt.replace("[DOC_NAME]", doc_name)
    # print(prompt)
    return prompt

def get_data_card_prompt(fields, doc, dataset_name, doc_name):
    with open(os.path.join(os.path.dirname(__file__), "prompts/data_card_prompt.txt"), "r") as text_file:
        prompt = text_file.read()

    fields_str = '\n'.join([f"{f[0]}: {f[1]}" for f in fields])
    prompt = prompt.replace("[FIELDS]", fields_str)
    prompt = prompt.replace("[DOC]", doc)
    prompt = prompt.replace("[DATASET_NAME]", dataset_name)
    prompt = prompt.replace("[DOC_NAME]", doc_name)
    return prompt

def get_model_card_prompt(fields, text, code):
    with open(os.path.join(os.path.dirname(__file__), "prompts/model_card_prompt.txt"), "r") as text_file:
        prompt = text_file.read()

    fields_str = '\n'.join([f"{f[0]}: {f[1]}" for f in fields])
    prompt = prompt.replace("[FIELDS]", fields_str)
    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[CODE]", code)
    return prompt


def get_text_column_prompt(text, column):
    text_file = open(os.path.join(os.path.dirname(__file__), "prompts/text_column_prompt.txt"), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[COLUMN]", column)

    return prompt



def get_variables(path):
    list = []
    with open(path) as myFile:
        for num, line in enumerate(myFile, 1):
            match = re.match(r'\s*(\S+)\s*=\s*([-+]?(?:\d*\.\d+|\d+))\s*', line)
            if match:
                para = match.group(1)
                val = match.group(2)
                # print(num, ",", para, ",", val)
                list.append((num, para, val))
    print("Extracted arizona-extraction: ", list)
    return list

def match_code_targets(targets, code_path, terms):
    vars = get_variables(code_path)
    vdict = {}
    connection = []
    for idx, v in enumerate(vars):
        vdict[v[1]] = idx
    for t in targets:
        val = get_match(vars, terms, t)
        connection.append((t, {val: "grometSubObject"}, float(vars[vdict[val]][2]), vars[vdict[val]][0]))
    return connection


def ontology_code_connection():
    terms = ['population', 'doubling time', 'recovery time', 'infectious time']
    code = "model/SIR/CHIME_SIR_while_loop.py"
    targets = ['population', 'infectious time']
    val = []
    try:
        val = match_code_targets(targets, code, terms)
    except OpenAIError as err:
        print("OpenAI connection error:", err)
        print("Using hard-coded connections")
        val = [("infectious time", {"name": "grometSubObject"}, 14.0, 67),
               ("population", {"name": "grometSubObject"}, 1000, 80)]
    print(val)



def code_text_connection(code, text, gpt_key, interactive = False):
    code_str = code
    idx_text = index_text(text)
    tlist = text.splitlines()
    targets = extract_func_names(code_str)
    print(f"TARGETS: {targets}")
    tups = []
    try:
        for t in targets:
            prompt = get_code_text_prompt(code_str, idx_text, t)
            match = get_gpt_match(prompt, gpt_key)
            ilist = extract_ints(match)
            ret_s = select_text(tlist, int(ilist[0]), int(ilist[-1]), 1, interactive)
            if interactive:
                print("Best description for python function {} is in lines {}-{}:".format(t, ilist[0], ilist[-1]))
                print(ret_s)
                print("---------------------------------------")
            else:
                tups.append((t, int(ilist[0]), int(ilist[-1]), ret_s))
        return tups, True
    except OpenAIError as err:
        if interactive:
            print("OpenAI connection error:", err)
        else:
            return f"OpenAI connection error: {err}", False


def code_dataset_connection(code, schema, gpt_key, interactive=False):
    targets = extract_func_names(code)
    tups = []
    try:
        for t in targets:
            prompt = get_code_dataset_prompt(code, schema, t)
            match = get_gpt_match(prompt, gpt_key)
            returnable = match.split("function are the ")[1].split(" columns.")[0].split(" and ")

            if interactive:
                print(returnable)
                print("---------------------------------------")
            else:
                tups.append((t, returnable))
        return tups, True
    except OpenAIError as err:
        if interactive:
            print("OpenAI connection error:", err)
        else:
            return f"OpenAI connection error: {err}",False

def text_column_connection(text, column, gpt_key):
    try:
        prompt = get_text_column_prompt(text, column)
        match = get_gpt_match(prompt, gpt_key)
        return match, True
    except OpenAIError as err:
        return f"OpenAI connection error: {err}",False


def rank_dkg_terms(target, concepts, gpt_key):
    """
    Rank the concepts by their similarity to the target
    :param target: Target concept json
    :param concepts: List of candidate concepts in json
    :return: List of ranked concepts
    """
    prompt = get_rank_dkg_prompt(json.dumps(target), json.dumps(concepts))
    match = get_gpt4_match(prompt, gpt_key, model="gpt-4")
    rank = match.splitlines()
    sorted = sort_dkg(rank, concepts)
    return sorted, True


def sort_dkg(ranking, json_obj):
    # Create a dictionary with the ranking as keys and the corresponding JSON objects as values
    ranking_dict = {item[0]: item for item in json_obj if item[0] in ranking}

    # Sort the dictionary based on the ranking list
    sorted_dict = {k: ranking_dict[k] for k in ranking if k in ranking_dict}

    # Convert the sorted dictionary back to a list of lists
    sorted_json = list(sorted_dict.values())

    return sorted_json


async def profile_matrix(data: List[List[Any]], doc, dataset_name, doc_name, gpt_key='', smart=False):
    """
    Grounding a matrix of data to DKG terms
    """
    if not data:
        return f"Empty dataset input", False

    if not all(all(isinstance(x, (int, float)) for x in row) for row in data):
        return f"Matrix data must be all-numeric. Data was: {data}", False

    # for matrices, we compute statistics across the entire matrix
    df = pd.DataFrame(data)
    df = df.stack()
    stats = {
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
        "quantile_25": df.quantile(0.25),
        "quantile_50": df.quantile(0.5),
        "quantile_75": df.quantile(0.75),
        "num_null_entries": int(df.isnull().sum()),
        "type": "numeric",
    }

    return json.dumps({'matrix_stats': stats}), True

def dataset_header_dkg(cols, gpt_key=''):
    """
    Grounding the column header to DKG terms
    :param cols: List of column names
    :return: Matches column name to DKG
    """
    col_ant = {}
    for col in cols:
        print(f"looking up {col}")
        results = []
        results.extend(get_mira_dkg_term(col, ['id', 'name'],True))
        # print(results)
        seen = set()
        for res in results:
            if not res:
                break
            seen.add(res[0])

        ans = get_gpt_match(f"What's the top 2 similar terms of \"{col}\" in epidemiology? Please list these terms separated by comma.", gpt_key, model="text-davinci-003")
        print(f"relevant items found from GPT: {ans}")
        for e in ans.split(","):
            # print(e)
            e = e.strip()
            for res in get_mira_dkg_term(e, ['id', 'name', 'type'],True):
                # print(res)
                if not res:
                    break
                if not res[0] in seen:
                    results.append(res)
                    seen.add(res[0])
        col_ant[col] = results
    return json.dumps(col_ant), True


from mitaskem.src.kgmatching import local_batch_get_mira_dkg_term

async def dataset_header_document_dkg(data, doc, dataset_name, doc_name, gpt_key='', smart=False):
    """
    Grounding a dataset to a DKG
    :param data: Dataset as a list of lists, including header and optionally a few rows
    :param doc: Document string
    :param dataset_name: Name of dataset
    :param doc_name: Name of document
    :param gpt_key: OpenAI API key
    :return: Matches column name to DKG
    """
    if not data:
        return f"Empty dataset input", False
    header = data[0]
    data = data[1:]
    schema = ','.join(header)

    print("Getting stats")
    stats = await _compute_tabular_statistics(data, header=header)

    col_ant = OrderedDict()

    prompt = get_csv_doc_prompt(schema, stats, doc, dataset_name, doc_name)
    match = get_gpt4_match(prompt, gpt_key, model="gpt-4")
    print("Got match")
    print(match)
    match = match.split('```')
    if len(match) == 1:
        match = match[0]
    else:
        match = match[1]
    results = [s.strip() for s in match.splitlines()]
    results = [s for s in results if s]
    results = [[x.strip() for x in s.split("|")] for s in results]
    results = [x for x in results if len(x) == 4]
    if len(results) != len(header):
        return f"Got different number of results ({len(results)}) than columns ({len(header)})", False

    for res, col in zip(results, header):
        col_ant[col] = {}
        col_ant[col]["col_name"] = res[0]
        col_ant[col]["concept"] = res[1]
        col_ant[col]["unit"] = res[2]
        col_ant[col]["description"] = res[3]

    col_names = list(col_ant.keys())
    col_concepts = [col_ant[col]["concept"] for col in col_ant]
    col_descriptions = [col_ant[col]["description"] for col in col_ant]

    terms = [ f'{col_name}: {col_concept} {col_description}' for (col_name, col_concept, col_description) in zip(col_names, col_concepts, col_descriptions) ]
    matches0 = local_batch_get_mira_dkg_term(terms)
    matches = [[[res['id'], res['name'], res['type']] for res in batch] for batch in matches0]
    # # line up coroutines
    # ops = [abatch_get_mira_dkg_term(col_names, ['id', 'name', 'type'], True),
    #        abatch_get_mira_dkg_term(col_concepts, ['id', 'name', 'type'], True),
    #        ]
    # # let them all finish
    # name_results, concept_results = await asyncio.gather(*ops)

    for col, match in zip(col_names, matches):
        col_ant[col]["dkg_groundings"] = match
        if smart:
            target = copy.deepcopy(col_ant[col])
            del target["dkg_groundings"]
            res=rank_dkg_terms(target, match, gpt_key)[0]
            col_ant[col]["dkg_groundings"] = res
            print(f"Smart grounding for {col}: {res}")

    for col in col_ant:
        col_ant[col]["column_stats"] = stats.get(col, {})

    return json.dumps(col_ant), True


async def _compute_tabular_statistics(data: List[List[Any]], header):
    """
    Compute summary statistics for a given tabular dataset.
    :param data: Dataset as a list of lists
    :return: Summary statistics as a dictionary
    """

    csv_df = pd.DataFrame(data, columns=header)
    null_counts = csv_df.isnull().sum(axis=0)

    # first handle numeric columns
    df = csv_df.describe()
    df.drop('count', inplace=True)  # we don't want the count row
    # NaN and inf are not json serialiazable, so we replace them with strings
    df.fillna('NaN', inplace=True)
    df.replace(float('nan'), 'NaN', inplace=True)  # better safe than sorry
    df.replace(float('inf'), 'inf', inplace=True)
    df.replace(float('-inf'), '-inf', inplace=True)
    res = df.to_dict()
    key_translations = {f"{x}%": f"quantile_{x}" for x in (25, 50, 75)}
    for col in res.keys():
        res[col]['type'] = 'numeric'
        for k in list(res[col].keys()):
            if k in key_translations:
                res[col][key_translations[k]] = res[col].pop(k)

    # try to infer date columns and convert them to datetime objects
    date_cols = set()
    df = csv_df.select_dtypes(include=['object'])
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            date_cols.add(col)
        except Exception:
            continue

    # then handle categorical columns, saving the top 10 most common values along with their counts
    # (also do this for dates)
    for col in df.columns:
        res[col] = {'type': 'categorical'}
        res[col]['most_common_entries'] = {}
        # get top <=10 most common values along with their counts
        counts = df[col].value_counts()
        for i in range(min(10, len(counts))):
            val = counts.index[i]
            if col in date_cols:
                val = val.isoformat()
            res[col]['most_common_entries'][val] = int(counts[i])
        # get number of unique entries
        res[col]['num_unique_entries'] = len(df[col].unique())

        if col in date_cols:
            res[col]['type'] = 'date'
            res[col]['earliest'] = df[col].min().isoformat()
            res[col]['latest'] = df[col].max().isoformat()

    for col in res:
        res[col]['num_null_entries'] = int(null_counts[col])

    # make sure all column indices are strings
    res = {str(k): v for k, v in res.items()}

    return res


async def construct_data_card(data_doc, dataset_name, doc_name, dataset_type, gpt_key='', model="gpt-3.5-turbo-16k"):
    """
    Constructing a data card for a given dataset and its description.
    :param data: Small dataset, including header and optionally a few rows
    :param data_doc: Document string
    :param gpt_key: OpenAI API key
    :param model: OpenAI model to use
    :return: Data card
    """

    fields = [("DESCRIPTION",  "Short description of the dataset (1-3 sentences)."),
              ("AUTHOR_NAME",  "Name of publishing institution or author."),
              ("AUTHOR_EMAIL", "Email address for the author of this dataset."),
              ("DATE",         "Date of publication of this dataset."),
              ("PROVENANCE",   "Short (1 sentence) description of how the data was collected."),
              ("SENSITIVITY",  "Is there any human-identifying information in the dataset?"),
              ("LICENSE",      "License for this dataset."),
    ]
    if dataset_type == 'no-header':
        # also want GPT to fill in the schema
        fields.append(("SCHEMA", "The dataset schema, as a comma-separated list of column names."))
    elif dataset_type == 'matrix':
        # instead of a schema, want to ask GPT to explain what a given (row, column) cell means
        fields.append(("CELL_INTERPRETATION", "A brief description of what a given cell in the matrix represents (i.e. how to interpret the value at a given a row/column pair)."))

    prompt = get_data_card_prompt(fields, data_doc, dataset_name, doc_name)
    match = get_gpt4_match(prompt, gpt_key, model=model)
    print(match)

    results = OrderedDict([(f[0], "UNKNOWN") for f in fields])
    for res in match.splitlines():
        if res == "":
            continue
        res = res.split(":", 1)
        if len(res) != 2:
            continue
        field, value = res
        field = field.strip()
        value = value.strip()
        for f in fields:
            if f[0] == field:
                results[field] = value
                break

    return json.dumps(results), True

async def construct_model_card(text, code,  gpt_key='', model="gpt-3.5-turbo-16k"):
    """
    Constructing a data card for a given dataset and its description.
    :param text: Model description (either model documentation or related paper)
    :param code: Model code
    :param gpt_key: OpenAI API key
    :param model: OpenAI model to use
    :return: Model card
    """

    fields = [("DESCRIPTION",  "Short description of the model (1 sentence)."),
              ("AUTHOR_INST",  "Name of publishing institution."),
              ("AUTHOR_AUTHOR", "Name of author(s)."),
              ("AUTHOR_EMAIL", "Email address for the author of this model."),
              ("DATE",         "Date of publication of this model."),
              ("SCHEMA",       "Short description of the schema of inputs and outputs of the model (1 sentence)."),
              ("PROVENANCE",   "Short description (1 sentence) of how the model was trained."),
              ("DATASET",      "Short description (1 sentence) of what dataset was used to train the model."),
              ("COMPLEXITY",  "The complexity of the model"),
              ("USAGE",        "Short description (1 sentence) of the context in which the model should be used"),
              ("LICENSE",      "License for this model."),
    ]

    prompt = get_model_card_prompt(fields, text, code)
    match = get_gpt4_match(prompt, gpt_key, model=model)
    print(match)

    results = OrderedDict([(f[0], "UNKNOWN") for f in fields])
    for res in match.splitlines():
        if res == "":
            continue
        res = res.split(":", 1)
        if len(res) != 2:
            continue
        field, value = res
        field = field.strip()
        value = value.strip()
        for f in fields:
            if f[0] == field:
                results[field] = value
                break

    return json.dumps(results), True


def select_text(lines, s, t, buffer, interactive=True):
    ret_s = ""
    start = s - buffer
    end = t + buffer
    if start < 0:
        start = 0
    if end >= len(lines):
        end = len(lines) - 1
    for i in range(start, end+1):
        if i<=t and i>=s:
            if interactive:
                ret_s += ">>\t{}\t{}".format(i,lines[i])
            else:
                ret_s += lines[i]
        elif interactive:
            ret_s += "\t{}\t{}".format(i, lines[i])
    return ret_s

def code_formula_connection(code, formulas, gpt_key):
    flist = formulas.splitlines()
    matches = []
    if flist[-1]=="":
        del flist[-1]
    try:
        for t in flist:
            prompt = get_code_formula_prompt(code, formulas, t)
            match = get_gpt_match(prompt, gpt_key)

            matches.append([t, match.split(":")[1]])
        return matches, True
    except OpenAIError as err:
        return f"OpenAI connection error: {err}", False

def represents_int(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def vars_dataset_connection_simplified(json_str, dataset_str, gpt_key):
    json_list = ast.literal_eval(json_str)

    var_list = list(filter(lambda x: x["type"] == "variable", json_list))

    all_desc_ls = [(var['id'] + ", " + var['name'] + ": " + var['text_annotations'][0]) for var in var_list]
    all_desc = '\n'.join(all_desc_ls)

    vs_data = {}

    try:
        prompt = get_var_dataset_prompt_simplified(all_desc, dataset_str)
        print(prompt)
        ans = get_gpt4_match(prompt, gpt_key, model="gpt-3.5-turbo-16k")
        ans = ans.splitlines()
        print(ans)
        for item in ans:
            toks = item.split(": ")
            if len(toks) < 2:
                continue
            vid = toks[0]
            cols = []
            for ds in toks[1].split(", "):
                data_col = ds.split("___")
                if len(data_col) < 2:
                    continue
                cols.append([data_col[0], data_col[1]])

            vs_data[vid] = cols
            print(cols)

    except OpenAIError as err:
        return f"OpenAI connection error: {err}", False

    for item in json_list:
        if item["type"] != "variable":
            continue

        id = item["id"]
        item["data_annotations"] = vs_data[id]

    new_json_str = json.dumps(json_list)

    return new_json_str, True


def vars_dataset_connection(json_str, dataset_str, gpt_key):
    json_list = ast.literal_eval(json_str)

    var_list = list(filter(lambda x: x["type"] == "variable", json_list))

    all_desc_ls = [(var['name']+": "+var['text_annotations'][0]) for var in var_list]
    all_desc = '\n'.join(all_desc_ls)

    vs_data = {}

    dataset_s = ""
    datasets = dataset_str.splitlines()
    dataset_name_dict = {}
    i = 0
    for d in tqdm(datasets):
        toks = d.split(":")
        if len(toks) != 2:
            continue
        name, cols = toks
        cols = cols.split(",")
        dataset_name_dict[i] = name
        for col in cols:
            dataset_s += str(i) + "___" + col.strip() + "\n"
        i += 1

    try:
        for i in tqdm(range(len(all_desc_ls))):
            prompt = get_var_dataset_prompt(all_desc, dataset_s, all_desc_ls[i])
            print(prompt)
            ans = get_gpt4_match(prompt, gpt_key, model="gpt-3.5-turbo")
            ans = ans.splitlines()
            print(ans)
            for j in range(len(ans)):
                toks = ans[j].split("___")
                # print(toks)
                if len(toks) < 2 or not represents_int(toks[0]):
                    ans[j] = ""
                else:
                    d_name = dataset_name_dict[int(toks[0])]
                    col_name = "___".join(toks[k] for k in range(1, len(toks)))
                    ans[j] = [d_name, col_name]

            vs_data[var_list[i]['id']] = ans
            time.sleep(5)
            # print(f"assigned value {ans} to key {var_list[i]['id']}")

    except OpenAIError as err:
        return f"OpenAI connection error: {err}", False

    for item in json_list:
        if item["type"] != "variable":
            continue

        id = item["id"]
        item["data_annotations"] = vs_data[id]

    # matches_str = ",".join(
    #     [("\"" + var + "\":[\"" + "\",\"".join([str(item) for item in vs_data[var]]) + "\"]") for var in
    #         vs_data])

    # s = ", {\"type\":\"datasetmap\""+ \
    #     ", \"id\":\"d" + str(hash(matches_str) % ((sys.maxsize + 1) * 2)) + \
    #     "\", \"matches\": " + json.dumps(vs_data) + " }]"

    # new_json_str = json_str[:-1] + s

    new_json_str = json.dumps(json_list)

    return new_json_str, True


def vars_formula_connection(json_str, formula, gpt_key):
    json_list = ast.literal_eval(json_str)

    var_list = list(filter(lambda x: x["type"] == "variable", json_list))

    prompt = get_formula_var_prompt(formula)
    latex_vars = get_gpt_match(prompt, gpt_key)
    latex_vars = latex_vars.split(':')[1].split(',')

    latex_var_set = {}

    all_desc_ls = [var['name'] for var in var_list]
    all_desc = '\n'.join(all_desc_ls)

    try:
        for latex_var in tqdm(latex_vars):
            prompt = get_all_desc_formula_prompt(all_desc, latex_var)
            ans = get_gpt_match(prompt, gpt_key)
            ans = ans.splitlines()

            matches = []
            for a in ans:
                if a in all_desc_ls:
                    a_idx = all_desc_ls.index(a)
                    matches.append(var_list[a_idx]['id'])
            latex_var_set[latex_var] = matches

            # for desc in tqdm(var_list):
            #     prompt = get_var_formula_prompt(desc["name"], latex_var)
            #     ans = get_gpt_match(prompt, gpt_key, model="text-davinci-003")
            #     ans = ans.split(':')[1]


            #     if ans == 'YES':
            #         current_matches = latex_var_set.get(latex_var, [])
            #         current_matches.append(desc["id"])
            #         latex_var_set[latex_var] = current_matches


        matches_str = ",".join([("\"" + var + "\" : [\"" + "\",\"".join([str(item) for item in latex_var_set[var]]) + "\"]") for var in latex_var_set])

        s = ", {\"type\":\"equation\", \"latex\":" + formula + \
            ", \"id\":\"e" + str(hash(formula)%((sys.maxsize + 1) * 2))+\
            "\", \"matches\": {" + matches_str + "} }]"

        new_json_str = json_str[:-1] + s

        return new_json_str, True
    except OpenAIError as err:
        return f"OpenAI connection error: {err}", False

DEFAULT_TERMS = ['population', 'doubling time', 'recovery time', 'infectious time']
DEFAULT_ATTRIBS = ['description', 'synonyms', 'xrefs', 'suggested_unit', 'suggested_data_type',
           'physical_min', 'physical_max', 'typical_min', 'typical_max']

def code_dkg_connection(dkg_targets, gpt_key, ontology_terms=DEFAULT_TERMS, ontology_attribs=DEFAULT_ATTRIBS):

    gromet_fn_module = json_to_gromet("gromet/CHIME_SIR_while_loop--Gromet-FN-auto.json")
    nops = query.collect_named_output_ports(gromet_fn_module)

    terms = list(build_local_ontology(ontology_terms, ontology_attribs).keys())
    variables = set()
    var_dict = {}
    for nop in nops:
        if nop[1] is not None:
            variables.add(nop[0])
            var_dict[nop[0]] = nop

    vlist = []
    for v in list(variables):
        vlist.append((var_dict[v][2].to_dict()['line_begin'], v, var_dict[v][1].to_dict()['value']))
    connection = []
    for t in dkg_targets:
        prompt = get_code_dkg_prompt(vlist, terms, t)
        match = get_gpt_match(prompt, gpt_key)
        val = match.split("(")[1].split(",")[0]

        connection.append((t, {val: "grometSubObject"}, float(var_dict[val][1].to_dict()['value']),
                           var_dict[val][2].to_dict()['line_begin']))

    print(connection)
    return connection


def _is_numeric(s):
    try:
        locale.atof(s)
        return True
    except ValueError:
        return False

def process_data(data: List[List[Any]]) -> List[List[Any]]:
    """
    Convert all numeric values in a dataset to floats, casting empty strings to NaNs.
    :param data: Dataset as a list of lists
    :return: Dataset with all numeric values converted to floats
    """
    def f(x):
        if x == '':
            return float('nan')
        elif _is_numeric(x):
            return locale.atof(x)
        else:
            return x

    return [[f(x) for x in row] for row in data]


def get_dataset_type(first_row: List[Any]) -> str:
    if all([_is_numeric(s) for s in first_row]):
        return 'matrix'
    elif any([_is_numeric(s) for s in first_row]):
        return 'no-header'
    else:
        return 'header-0'


if __name__ == "__main__":

    dkg_str = """{
      "col_name": "dates",
      "concept": "Time",
      "unit": "YYYY-MM-DD",
      "description": "The date when the data was recorded",
      "dkg_groundings": [
        [
          "hp:0001147",
          "Retinal exudate"
        ],
        [
          "hp:0030496",
          "Macular exudate"
        ],
        [
          "ncit:C114947",
          "Postterm Infant"
        ],
        [
          "oae:0006126",
          "retinal exudates AE"
        ],
        [
          "oae:0008293",
          "large for dates baby AE"
        ],
        [
          "pato:0000165",
          "time",
          "class"
        ],
        [
          "gfo:Time",
          "time",
          "class"
        ],
        [
          "geonames:2365173",
          "Maritime",
          "individual"
        ],
        [
          "wikidata:Q174728",
          "centimeter",
          "class"
        ],
        [
          "probonto:k0000056",
          "nondecisionTime",
          "class"
        ]
      ]
    }"""

    json_obj = json.loads(dkg_str)
    target = copy.deepcopy(json_obj)
    del target["dkg_groundings"]


    print(rank_dkg_terms(target, json_obj["dkg_groundings"], GPT_KEY))

    # code_dkg_connection("population", "") # GPT key
    # vars = read_text_from_file('../demos/2023-03-19/mar_demo_intermediate.json')
    # dataset = read_text_from_file('../resources/dataset/headers.txt')
    # match, _ = vars_dataset_connection(vars, dataset, GPT_KEY)
    # print(match)
    #
    # res, yes = dataset_header_document_dkg("""dates,VAX_count,day,sdm,events,I_1,I_2,I_3,Y_1,Y_2,Y_3,V_1,V_2,V_3,Infected,Y,V,logV""",
    #                                        """Using wastewater surveillance as a continuous pooled sampling technique has been in place in many countries since the early stages of the outbreak of COVID-19. Since the beginning of the outbreak, many research works have emerged, studying different aspects of *viral SARS-CoV-2 DNA concentrations* (viral load) in wastewater and its potential as an early warning method. However, one of the questions that has remained unanswered is the quantitative relation between viral load and clinical indicators such as daily cases, deaths, and hospitalizations. Few studies have tried to couple viral load data with an epidemiological model to relate the number of infections in the community to the viral burden. We propose a **stochastic wastewater-based SEIR model** to showcase the importance of viral load in the early detection and prediction of an outbreak in a community. We built three models based on whether or not they use the case count and viral load data and compared their *simulations* and *forecasting* quality.  We consider a stochastic wastewater-based epidemiological model with four compartments (hidden states) of susceptible (S), exposed (E), infectious (I), and recovered/removed (R).dRxiv} } """,GPT_KEY)
    # print(res)

    # col = "people"
    # ans = get_gpt_match(
    #     f"What's the top 2 similar terms of \"{col}\" in epidemiology? Please list these terms separated by comma.",
    #     GPT_KEY, model="text-davinci-003")
    # print(ans)
    # for latex_var in match:
    #     print(latex_var, match[latex_var])
    #     print('\n')